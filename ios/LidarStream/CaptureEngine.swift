import ARKit
import CoreGraphics
import CoreImage
import CoreVideo
import Foundation
import ImageIO      // kCGImageDestinationLossyCompressionQuality
import QuartzCore   // CACurrentMediaTime
import simd

/// One frame reduced to exactly what goes on the wire.
///
/// Carries depth only when the device actually has a LiDAR scanner. Without one the
/// depth payload is nil and the desktop bridge infers it from `jpeg` on the GPU --
/// either way the viewer receives depth + intrinsics + pose and cannot tell.
struct DepthFrame {
    let timestamp: Double
    let source: Source

    /// Depth map, LiDAR only.
    let depthWidth: Int
    let depthHeight: Int
    let depthMM: Data?       // UInt16 little-endian millimetres, 0 = no return
    let confidence: Data?    // UInt8 0..2

    let jpeg: Data?
    let jpegWidth: Int
    let jpegHeight: Int

    /// Pinhole intrinsics, scaled to the depth map in `.lidar` mode and to the JPEG
    /// in `.rgb` mode -- i.e. always to whatever image the receiver will unproject.
    let fx: Float
    let fy: Float
    let cx: Float
    let cy: Float

    let cameraToWorld: simd_float4x4
    let tracking: String

    enum Source: String {
        case lidar          // sensor depth, from a Pro-class device
        case rgb            // colour + pose only; depth is inferred on the desktop
    }
}

/// Drives ARKit and hands frames to whoever set `onFrame`.
///
/// World tracking works on any modern iPhone, so the 6DoF pose is always real. Only
/// the depth map needs a scanner, which is what `support` distinguishes.
final class CaptureEngine: NSObject, ObservableObject, ARSessionDelegate {

    enum Support: Equatable {
        case lidar              // has a scanner: real sensor depth
        case worldTrackingOnly  // no scanner: stream colour + pose, infer depth on the PC
        case noARKit
    }

    @Published private(set) var support: Support = .noARKit
    @Published private(set) var running = false
    @Published private(set) var tracking = "—"
    @Published private(set) var frameSize = "—"
    @Published private(set) var captureFPS: Double = 0
    @Published private(set) var preview: CGImage?

    @Published var useSmoothedDepth = true
    @Published var targetFPS: Double = 20
    @Published var jpegQuality: Double = 0.55

    /// Colour is optional on a LiDAR device but mandatory without one, since it is
    /// the only thing the depth model has to work from.
    @Published var sendColor = true

    /// Longest edge of the transmitted JPEG in `.rgb` mode. The depth model wants
    /// real detail here, unlike the LiDAR path where colour is only for tinting
    /// points and can match the 256x192 depth map.
    @Published var rgbLongEdge: Double = 640

    var onFrame: ((DepthFrame) -> Void)?

    private let session = ARSession()
    private let queue = DispatchQueue(label: "lidar.capture", qos: .userInitiated)
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    private var lastEmit = 0.0
    private var lastPreview = 0.0
    private var fpsWindowStart = 0.0
    private var fpsWindowCount = 0

    override init() {
        super.init()
        session.delegate = self
        session.delegateQueue = queue
        support = Self.detectSupport()
    }

    private static func detectSupport() -> Support {
        guard ARWorldTrackingConfiguration.isSupported else { return .noARKit }
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) ||
           ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) {
            return .lidar
        }
        return .worldTrackingOnly
    }

    var mode: DepthFrame.Source {
        support == .lidar ? .lidar : .rgb
    }

    func start() {
        guard support != .noARKit else { return }
        let config = ARWorldTrackingConfiguration()
        // Gravity-aligned so world +y is up and reconstructions come out upright.
        config.worldAlignment = .gravity
        config.environmentTexturing = .none
        config.planeDetection = []

        if support == .lidar {
            if useSmoothedDepth,
               ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) {
                config.frameSemantics.insert(.smoothedSceneDepth)
            } else {
                config.frameSemantics.insert(.sceneDepth)
            }
        }

        session.run(config, options: [.resetTracking, .removeExistingAnchors])
        DispatchQueue.main.async { self.running = true }
    }

    func stop() {
        session.pause()
        DispatchQueue.main.async {
            self.running = false
            self.tracking = "—"
            self.captureFPS = 0
        }
    }

    func restart() {
        guard running else { return }
        stop()
        start()
    }

    // MARK: - ARSessionDelegate

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let now = CACurrentMediaTime()
        guard now - lastEmit >= 1.0 / max(targetFPS, 1) - 0.002 else { return }

        let trackingText = Self.describe(frame.camera.trackingState)
        let res = frame.camera.imageResolution
        let K = frame.camera.intrinsics

        let out: DepthFrame?
        if support == .lidar {
            out = lidarFrame(frame, res: res, K: K, tracking: trackingText)
        } else {
            out = rgbFrame(frame, res: res, K: K, tracking: trackingText)
        }
        guard let out else { return }

        lastEmit = now
        onFrame?(out)

        fpsWindowCount += 1
        if fpsWindowStart == 0 { fpsWindowStart = now }
        let elapsed = now - fpsWindowStart
        if elapsed >= 1.0 {
            let fps = Double(fpsWindowCount) / elapsed
            let label = out.source == .lidar
                ? "\(out.depthWidth)x\(out.depthHeight) depth"
                : "\(out.jpegWidth)x\(out.jpegHeight) rgb"
            fpsWindowStart = now
            fpsWindowCount = 0
            DispatchQueue.main.async {
                self.captureFPS = fps
                self.tracking = trackingText
                self.frameSize = label
            }
        }

        if now - lastPreview > 0.12 {
            lastPreview = now
            if let mm = out.depthMM,
               let img = Self.depthPreview(mm, width: out.depthWidth,
                                           height: out.depthHeight) {
                DispatchQueue.main.async { self.preview = img }
            } else if out.source == .rgb, let jpeg = out.jpeg {
                // No depth to visualise, so show what is actually being sent.
                if let img = Self.cgImage(fromJPEG: jpeg) {
                    DispatchQueue.main.async { self.preview = img }
                }
            }
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        DispatchQueue.main.async {
            self.running = false
            self.tracking = "failed: \(error.localizedDescription)"
        }
    }

    func sessionWasInterrupted(_ session: ARSession) {
        DispatchQueue.main.async { self.tracking = "interrupted" }
    }

    func sessionInterruptionEnded(_ session: ARSession) {
        // Pose continuity is gone after an interruption; reset so the world frame
        // does not silently jump mid-scan.
        if running { start() }
    }

    // MARK: - frame builders

    private func lidarFrame(_ frame: ARFrame, res: CGSize, K: simd_float3x3,
                            tracking: String) -> DepthFrame? {
        guard let depth = useSmoothedDepth
                ? (frame.smoothedSceneDepth ?? frame.sceneDepth)
                : frame.sceneDepth
        else { return nil }

        let map = depth.depthMap
        let w = CVPixelBufferGetWidth(map)
        let h = CVPixelBufferGetHeight(map)
        guard let mm = Self.depthToMillimetres(map) else { return nil }
        let conf = depth.confidenceMap.flatMap { Self.copyUInt8Plane($0) }

        // ARKit intrinsics describe the full captured image; the depth map is a
        // smaller, aligned version of the same view, so scale them to match.
        let sx = Float(w) / Float(res.width)
        let sy = Float(h) / Float(res.height)

        var jpeg: Data?
        if sendColor {
            jpeg = encodeJPEG(frame.capturedImage, width: w, height: h)
        }

        return DepthFrame(
            timestamp: frame.timestamp, source: .lidar,
            depthWidth: w, depthHeight: h, depthMM: mm, confidence: conf,
            jpeg: jpeg, jpegWidth: w, jpegHeight: h,
            fx: K[0][0] * sx, fy: K[1][1] * sy,
            cx: K[2][0] * sx, cy: K[2][1] * sy,
            cameraToWorld: frame.camera.transform, tracking: tracking)
    }

    /// No scanner: send colour plus the real 6DoF pose and let the desktop infer
    /// depth. Intrinsics are scaled to the transmitted JPEG, which is what the
    /// bridge unprojects against.
    private func rgbFrame(_ frame: ARFrame, res: CGSize, K: simd_float3x3,
                          tracking: String) -> DepthFrame? {
        let long = max(res.width, res.height)
        guard long > 0 else { return nil }
        let scale = min(CGFloat(rgbLongEdge) / long, 1.0)
        let w = Int((res.width * scale).rounded())
        let h = Int((res.height * scale).rounded())

        guard let jpeg = encodeJPEG(frame.capturedImage, width: w, height: h) else {
            return nil
        }

        let sx = Float(w) / Float(res.width)
        let sy = Float(h) / Float(res.height)

        return DepthFrame(
            timestamp: frame.timestamp, source: .rgb,
            depthWidth: 0, depthHeight: 0, depthMM: nil, confidence: nil,
            jpeg: jpeg, jpegWidth: w, jpegHeight: h,
            fx: K[0][0] * sx, fy: K[1][1] * sy,
            cx: K[2][0] * sx, cy: K[2][1] * sy,
            cameraToWorld: frame.camera.transform, tracking: tracking)
    }

    // MARK: - pixel buffer conversion

    /// Float32 metres -> UInt16 little-endian millimetres. Halves the payload for
    /// precision loss (1 mm) far below the sensor's own noise floor.
    private static func depthToMillimetres(_ pb: CVPixelBuffer) -> Data? {
        guard CVPixelBufferGetPixelFormatType(pb) == kCVPixelFormatType_DepthFloat32
        else { return nil }
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return nil }

        let w = CVPixelBufferGetWidth(pb)
        let h = CVPixelBufferGetHeight(pb)
        let rowBytes = CVPixelBufferGetBytesPerRow(pb)

        var out = Data(count: w * h * 2)
        out.withUnsafeMutableBytes { raw in
            guard let dst = raw.baseAddress?.assumingMemoryBound(to: UInt16.self) else { return }
            for y in 0..<h {
                let row = base.advanced(by: y * rowBytes)
                              .assumingMemoryBound(to: Float32.self)
                let outRow = dst.advanced(by: y * w)
                for x in 0..<w {
                    let m = row[x]
                    outRow[x] = (m.isFinite && m > 0) ? UInt16(min(m * 1000.0, 65535.0)) : 0
                }
            }
        }
        return out
    }

    private static func copyUInt8Plane(_ pb: CVPixelBuffer) -> Data? {
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return nil }
        let w = CVPixelBufferGetWidth(pb)
        let h = CVPixelBufferGetHeight(pb)
        let rowBytes = CVPixelBufferGetBytesPerRow(pb)

        // Row by row: the buffer is usually padded, and the receiver expects exactly
        // w*h tightly packed bytes.
        var out = Data(count: w * h)
        out.withUnsafeMutableBytes { raw in
            guard let dst = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
            for y in 0..<h {
                memcpy(dst.advanced(by: y * w), base.advanced(by: y * rowBytes), w)
            }
        }
        return out
    }

    /// Scale the captured frame to `width`x`height` and JPEG it. The camera image is
    /// the same view as the depth map, so a uniform scale keeps them aligned.
    private func encodeJPEG(_ pb: CVPixelBuffer, width: Int, height: Int) -> Data? {
        let ci = CIImage(cvPixelBuffer: pb)
        guard ci.extent.width > 0, width > 0, height > 0 else { return nil }
        let scale = CGFloat(width) / ci.extent.width
        let scaled = ci.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        let target = CGRect(x: 0, y: 0, width: CGFloat(width), height: CGFloat(height))
        return ciContext.jpegRepresentation(
            of: scaled.cropped(to: target),
            colorSpace: CGColorSpaceCreateDeviceRGB(),
            options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption:
                        jpegQuality])
    }

    private static func cgImage(fromJPEG data: Data) -> CGImage? {
        guard let src = CGImageSourceCreateWithData(data as CFData, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(src, 0, nil)
    }

    /// Small on-device depth visualisation, so it is obvious the sensor is alive even
    /// when the desktop viewer is not connected.
    private static func depthPreview(_ depthMM: Data, width: Int, height: Int) -> CGImage? {
        guard width > 0, height > 0 else { return nil }
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        depthMM.withUnsafeBytes { raw in
            let src = raw.bindMemory(to: UInt16.self)
            for i in 0..<min(width * height, src.count) {
                let mm = src[i]
                if mm == 0 {
                    rgba[i*4] = 12; rgba[i*4+1] = 14; rgba[i*4+2] = 18; rgba[i*4+3] = 255
                    continue
                }
                let t = min(max((Float(mm) / 1000.0 - 0.2) / 4.8, 0), 1)   // 0.2..5 m
                let (r, g, b) = turbo(t)
                rgba[i*4] = r; rgba[i*4+1] = g; rgba[i*4+2] = b; rgba[i*4+3] = 255
            }
        }
        guard let provider = CGDataProvider(data: Data(rgba) as CFData) else { return nil }
        return CGImage(width: width, height: height,
                       bitsPerComponent: 8, bitsPerPixel: 32,
                       bytesPerRow: width * 4,
                       space: CGColorSpaceCreateDeviceRGB(),
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                       provider: provider, decode: nil,
                       shouldInterpolate: false, intent: .defaultIntent)
    }

    /// Same Turbo ramp the desktop viewer uses, so the two read alike.
    private static func turbo(_ x: Float) -> (UInt8, UInt8, UInt8) {
        let r = 0.13572138 + x * (4.61539260 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943))))
        let g = 0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604))))
        let b = 0.10667330 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973))))
        func c(_ v: Float) -> UInt8 { UInt8(min(max(v, 0), 1) * 255) }
        return (c(r), c(g), c(b))
    }

    private static func describe(_ s: ARCamera.TrackingState) -> String {
        switch s {
        case .normal: return "normal"
        case .notAvailable: return "notAvailable"
        case .limited(let reason):
            switch reason {
            case .initializing: return "limited/initializing"
            case .excessiveMotion: return "limited/motion"
            case .insufficientFeatures: return "limited/features"
            case .relocalizing: return "limited/relocalizing"
            @unknown default: return "limited"
            }
        @unknown default: return "unknown"
        }
    }
}
