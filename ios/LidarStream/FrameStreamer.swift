import Compression
import Foundation
import Network
import simd

/// Ships `DepthFrame`s to the desktop bridge over plain TCP.
///
/// Wire format per frame (little-endian lengths):
///
///     "LFRM" u32 headerLen headerJSON depth conf rgb
///
/// `depth` and `conf` are raw DEFLATE (RFC 1951), which is exactly what Apple's
/// Compression framework produces for `COMPRESSION_ZLIB` -- no third-party zlib
/// needed on this side. The bridge inflates with wbits = -15.
final class FrameStreamer: ObservableObject {

    enum Link: Equatable {
        case idle
        case connecting
        case live
        case failed(String)

        var text: String {
            switch self {
            case .idle: return "not connected"
            case .connecting: return "connecting…"
            case .live: return "streaming"
            case .failed(let m): return m
            }
        }
    }

    @Published private(set) var link: Link = .idle
    @Published private(set) var framesSent = 0
    @Published private(set) var framesDropped = 0
    @Published private(set) var kbps: Double = 0

    private let queue = DispatchQueue(label: "lidar.net")

    /// `send` runs on the capture queue while `connect`/`disconnect` run on main and
    /// completions run on `queue`, so the connection and its backlog counter live
    /// behind a lock rather than being read across queues.
    private let lock = NSLock()
    private var conn: NWConnection?     // guarded by lock
    private var isLive = false          // guarded by lock
    private var inFlight = 0            // guarded by lock

    /// Depth capture must never outrun the link, or latency grows without bound
    /// while memory fills with frames that are already stale.
    private let maxInFlight = 2

    private var windowStart = Date()
    private var windowBytes = 0

    private struct Header: Encodable {
        let t: Double
        /// Depth dimensions, or 0 when there is no scanner and the bridge will infer
        /// depth from the JPEG. The intrinsics below always describe whichever image
        /// the receiver unprojects, so it never has to guess which one they match.
        let dw: Int
        let dh: Int
        let iw: Int
        let ih: Int
        let fx: Float
        let fy: Float
        let cx: Float
        let cy: Float
        let view: [Float]
        let src: String
        let track: String
        let dlen: Int
        let clen: Int
        let rlen: Int
    }

    // MARK: - connection

    func connect(host: String, port: UInt16, deviceName: String) {
        disconnect()

        guard let nwPort = NWEndpoint.Port(rawValue: port) else {
            setLink(.failed("bad port"))
            return
        }

        let params = NWParameters.tcp
        if let tcp = params.defaultProtocolStack.transportProtocol as? NWProtocolTCP.Options {
            tcp.noDelay = true                  // depth frames are latency-sensitive
            tcp.connectionTimeout = 8
        }

        let c = NWConnection(host: NWEndpoint.Host(host), port: nwPort, using: params)
        lock.lock()
        conn = c
        isLive = false
        inFlight = 0
        lock.unlock()
        setLink(.connecting)

        c.stateUpdateHandler = { [weak self] state in
            guard let self else { return }
            switch state {
            case .ready:
                self.sendHello(deviceName: deviceName, on: c)
                self.lock.lock(); self.isLive = true; self.lock.unlock()
                self.setLink(.live)
            case .waiting(let err):
                // Most often the Local Network permission prompt has not been
                // accepted yet, or nothing is listening on that address.
                self.setLink(.failed("waiting: \(err.localizedDescription)"))
            case .failed(let err):
                self.setLink(.failed(err.localizedDescription))
                self.teardown()
            case .cancelled:
                self.teardown()
                self.setLink(.idle)
            default:
                break
            }
        }
        c.start(queue: queue)
    }

    func disconnect() {
        lock.lock()
        let c = conn
        conn = nil
        isLive = false
        inFlight = 0
        lock.unlock()
        c?.cancel()
        setLink(.idle)
    }

    private func teardown() {
        lock.lock()
        conn = nil
        isLive = false
        inFlight = 0
        lock.unlock()
    }

    private func setLink(_ l: Link) {
        DispatchQueue.main.async { self.link = l }
    }

    private func sendHello(deviceName: String, on c: NWConnection) {
        let info: [String: Any] = [
            "name": deviceName,
            "model": deviceModelIdentifier(),
            "src": "lidar",
            "app": "LidarStream",
        ]
        guard let json = try? JSONSerialization.data(withJSONObject: info) else { return }
        var out = Data("LIDRHELO".utf8)
        out.append(le32(UInt32(json.count)))
        out.append(json)
        c.send(content: out, completion: .contentProcessed { _ in })
    }

    // MARK: - frames

    /// Called on the capture queue. Returns immediately; drops the frame if the
    /// socket is already backed up.
    func send(_ frame: DepthFrame) {
        lock.lock()
        guard let c = conn, isLive else { lock.unlock(); return }
        if inFlight >= maxInFlight {
            lock.unlock()
            DispatchQueue.main.async { self.framesDropped += 1 }
            return
        }
        inFlight += 1
        lock.unlock()

        let depth = frame.depthMM.flatMap { self.rawDeflate($0) } ?? Data()
        let conf = frame.confidence.flatMap { self.rawDeflate($0) } ?? Data()
        let rgb = frame.jpeg ?? Data()

        // A frame with neither depth nor colour carries nothing the viewer can use,
        // and without colour the bridge has nothing to infer depth from either.
        if depth.isEmpty && rgb.isEmpty {
            releaseSlot()
            return
        }

        var view: [Float] = []
        view.reserveCapacity(16)
        for col in 0..<4 {
            let v = frame.cameraToWorld[col]
            view.append(v.x); view.append(v.y); view.append(v.z); view.append(v.w)
        }

        let header = Header(
            t: frame.timestamp,
            dw: frame.depthWidth, dh: frame.depthHeight,
            iw: frame.jpegWidth, ih: frame.jpegHeight,
            fx: frame.fx, fy: frame.fy, cx: frame.cx, cy: frame.cy,
            view: view,
            src: frame.source.rawValue, track: frame.tracking,
            dlen: depth.count, clen: conf.count, rlen: rgb.count
        )
        guard let hb = try? JSONEncoder().encode(header) else {
            releaseSlot()
            return
        }

        var out = Data(capacity: 8 + hb.count + depth.count + conf.count + rgb.count)
        out.append(Data("LFRM".utf8))
        out.append(le32(UInt32(hb.count)))
        out.append(hb)
        out.append(depth)
        out.append(conf)
        out.append(rgb)

        let nbytes = out.count
        c.send(content: out, completion: .contentProcessed { [weak self] error in
            guard let self else { return }
            self.releaseSlot()
            if let error {
                self.setLink(.failed(error.localizedDescription))
                return
            }
            DispatchQueue.main.async {
                self.framesSent += 1
                self.windowBytes += nbytes
                let dt = Date().timeIntervalSince(self.windowStart)
                if dt >= 1.0 {
                    self.kbps = Double(self.windowBytes) / dt / 1024.0
                    self.windowBytes = 0
                    self.windowStart = Date()
                }
            }
        })
    }

    // MARK: - helpers

    private func releaseSlot() {
        lock.lock()
        inFlight = max(0, inFlight - 1)
        lock.unlock()
    }

    private func le32(_ v: UInt32) -> Data {
        var le = v.littleEndian
        return withUnsafeBytes(of: &le) { Data($0) }
    }

    /// Raw DEFLATE, no zlib or gzip wrapper. `COMPRESSION_ZLIB` is Apple's name for
    /// RFC 1951, which is the headerless form.
    private func rawDeflate(_ input: Data) -> Data? {
        guard !input.isEmpty else { return nil }
        // Generous ceiling: DEFLATE's incompressible worst case is the input plus a
        // few bytes of block overhead per 64 KiB.
        let cap = input.count + input.count / 64 + 128
        var out = Data(count: cap)
        let written: Int = out.withUnsafeMutableBytes { dstRaw in
            input.withUnsafeBytes { srcRaw in
                guard let dst = dstRaw.baseAddress?.assumingMemoryBound(to: UInt8.self),
                      let src = srcRaw.baseAddress?.assumingMemoryBound(to: UInt8.self)
                else { return 0 }
                return compression_encode_buffer(dst, cap, src, input.count, nil,
                                                 COMPRESSION_ZLIB)
            }
        }
        guard written > 0 else { return nil }
        return Data(out.prefix(written))
    }
}

/// e.g. "iPhone15,2" -- more useful in the viewer's HUD than the user-set name.
/// The bytes are copied out inside the closure; a pointer into `sysinfo` must not
/// escape it.
func deviceModelIdentifier() -> String {
    var sysinfo = utsname()
    guard uname(&sysinfo) == 0 else { return "unknown" }
    return withUnsafeBytes(of: &sysinfo.machine) { raw -> String in
        let bytes = Array(raw.bindMemory(to: UInt8.self).prefix { $0 != 0 })
        return String(decoding: bytes, as: UTF8.self)
    }
}
