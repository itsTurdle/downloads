import SwiftUI
import UIKit

struct ContentView: View {
    @StateObject private var engine = CaptureEngine()
    @StateObject private var net = FrameStreamer()

    @AppStorage("lidar.host") private var host = ""
    @AppStorage("lidar.port") private var portText = "8771"

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 18) {
                    preview
                    switch engine.support {
                    case .lidar, .worldTrackingOnly:
                        if engine.support == .worldTrackingOnly { noScannerNote }
                        controls
                    case .noARKit:
                        unsupported("ARKit unavailable",
                                    "This device cannot run world tracking.")
                    }
                }
                .padding(18)
            }
            .background(Color(red: 0.027, green: 0.031, blue: 0.043).ignoresSafeArea())
            .navigationTitle("LidarStream")
        }
        .onAppear {
            // The capture engine hands every frame straight to the socket; the
            // streamer drops it if the link is behind.
            engine.onFrame = { [weak net] frame in net?.send(frame) }
            UIApplication.shared.isIdleTimerDisabled = true
            // World tracking runs with or without a scanner, so start for both.
            if engine.support != .noARKit { engine.start() }
        }
        .onDisappear {
            UIApplication.shared.isIdleTimerDisabled = false
            engine.stop()
            net.disconnect()
        }
    }

    // MARK: - pieces

    private var preview: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 14)
                .fill(Color.white.opacity(0.04))
            if let img = engine.preview {
                Image(decorative: img, scale: 1, orientation: .up)
                    .resizable()
                    .interpolation(.none)
                    .aspectRatio(contentMode: .fit)
                    .clipShape(RoundedRectangle(cornerRadius: 14))
            } else {
                VStack(spacing: 8) {
                    Image(systemName: "cube.transparent")
                        .font(.system(size: 34))
                        .foregroundStyle(.secondary)
                    Text(engine.running ? "waiting for depth…" : "capture stopped")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .aspectRatio(4.0/3.0, contentMode: .fit)
        .overlay(alignment: .topLeading) {
            Text(net.link.text.uppercased())
                .font(.system(size: 10, weight: .semibold, design: .monospaced))
                .padding(.horizontal, 8).padding(.vertical, 4)
                .background(linkColor.opacity(0.22), in: Capsule())
                .foregroundStyle(linkColor)
                .padding(10)
        }
    }

    private var linkColor: Color {
        switch net.link {
        case .live: return .green
        case .connecting: return .orange
        case .failed: return .red
        case .idle: return .gray
        }
    }

    private var controls: some View {
        VStack(spacing: 16) {
            card("Viewer") {
                HStack(spacing: 10) {
                    TextField("192.168.1.20", text: $host)
                        .textFieldStyle(.roundedBorder)
                        .keyboardType(.numbersAndPunctuation)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                    TextField("8771", text: $portText)
                        .textFieldStyle(.roundedBorder)
                        .keyboardType(.numberPad)
                        .frame(width: 78)
                }
                Text("The address shown on the viewer page.")
                    .font(.caption2).foregroundStyle(.secondary)

                HStack(spacing: 10) {
                    Button {
                        let port = UInt16(portText) ?? 8771
                        net.connect(host: host.trimmingCharacters(in: .whitespaces),
                                    port: port,
                                    deviceName: UIDevice.current.name)
                    } label: {
                        Label("Connect", systemImage: "wifi")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(host.isEmpty || net.link == .connecting)

                    Button {
                        net.disconnect()
                    } label: {
                        Label("Stop", systemImage: "stop.fill")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                }
            }

            card("Stream") {
                stat("sending", engine.frameSize)
                stat("tracking", engine.tracking)
                stat("capture", String(format: "%.1f fps", engine.captureFPS))
                stat("sent", "\(net.framesSent)")
                stat("dropped", "\(net.framesDropped)")
                stat("uplink", String(format: "%.0f KiB/s", net.kbps))
            }

            card("Capture") {
                slider("rate", value: $engine.targetFPS, range: 5...30, step: 1,
                       format: { String(format: "%.0f fps", $0) })
                slider("jpeg quality", value: $engine.jpegQuality, range: 0.2...0.9, step: 0.05,
                       format: { String(format: "%.2f", $0) })

                if engine.support == .lidar {
                    Toggle("Send camera colour", isOn: $engine.sendColor)
                    Toggle("Smoothed depth", isOn: $engine.useSmoothedDepth)
                        .onChange(of: engine.useSmoothedDepth) { _, _ in engine.restart() }
                } else {
                    // Colour is the only input the depth model has, so it is not
                    // optional here, and its resolution drives depth detail.
                    slider("image size", value: $engine.rgbLongEdge,
                           range: 320...960, step: 32,
                           format: { String(format: "%.0f px", $0) })
                }

                HStack(spacing: 10) {
                    Button(engine.running ? "Pause capture" : "Start capture") {
                        if engine.running { engine.stop() } else { engine.start() }
                    }
                    .buttonStyle(.bordered)
                    .frame(maxWidth: .infinity)
                }
                .padding(.top, 2)
            }

            Text("Keep this screen open — iOS suspends capture in the background.")
                .font(.caption2)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
    }

    private var noScannerNote: some View {
        card("No LiDAR on this device") {
            Text("\(deviceModelIdentifier()) has no rear LiDAR scanner, so ARKit "
                 + "cannot supply a depth map. Streaming the camera plus real 6DoF "
                 + "pose instead — the desktop infers depth on the GPU.")
                .font(.callout)
                .foregroundStyle(.secondary)
        }
    }

    private func unsupported(_ title: String, _ body: String) -> some View {
        card(title) {
            Text(body).font(.callout).foregroundStyle(.secondary)
            Text("Model: \(deviceModelIdentifier())")
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(.tertiary)
        }
    }

    // MARK: - small builders

    private func card<C: View>(_ title: String, @ViewBuilder _ content: () -> C) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(title.uppercased())
                .font(.system(size: 10, weight: .semibold, design: .monospaced))
                .foregroundStyle(.secondary)
                .kerning(1.4)
            content()
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(14)
        .background(Color.white.opacity(0.05), in: RoundedRectangle(cornerRadius: 12))
    }

    private func stat(_ k: String, _ v: String) -> some View {
        HStack {
            Text(k).foregroundStyle(.secondary)
            Spacer()
            Text(v).monospacedDigit()
        }
        .font(.system(size: 13, design: .monospaced))
    }

    private func slider(_ label: String, value: Binding<Double>,
                        range: ClosedRange<Double>, step: Double,
                        format: @escaping (Double) -> String) -> some View {
        VStack(spacing: 2) {
            HStack {
                Text(label).foregroundStyle(.secondary)
                Spacer()
                Text(format(value.wrappedValue)).monospacedDigit()
            }
            .font(.system(size: 12, design: .monospaced))
            Slider(value: value, in: range, step: step)
        }
    }
}
