# iPhone → PC live 3D

Streams an iPhone's rear camera to this machine over WiFi and renders it as a live 3D
point cloud, fusing frames into a world-space scan you can export as `.ply`.

Three ways in, one display. The viewer only ever sees depth + intrinsics + pose, so it
cannot tell which path produced a frame.

```
  Safari, no install          native app,             native app,
  (capture.html)              no LiDAR                LiDAR (Pro)
  camera + orientation        camera + 6DoF pose      depth + colour + pose
        │ WSS 8443                  │ TCP 7771              │ TCP 7771
        ▼                           ▼                       ▼
   ┌──────────────────── lidar_server.py ─────────────────────┐
   │  Depth Anything V2 on the GPU fills in missing depth     │
   └───────────────────────────┬──────────────────────────────┘
                   HTTP 8770 / HTTPS 8443 · WS 8772 / WSS 8444
                               ▼
        browser: WebGL2 point cloud, fused scan, .ply export
```

## Quickest path: no app install at all

The native app needs a macOS runner to compile, and GitHub Actions is currently locked
account-wide over billing — so start with the browser instead.

```bash
cd server && python lidar_server.py
```

It prints a `https://<ip>:8443/capture.html` line. Open that **on the phone**, tap
*Start streaming*, and allow the camera. Then open <http://localhost:8770> on the PC to
watch the cloud.

### The certificate is not optional

iOS grants camera access only on a secure origin, and — the part that trips everyone up
— **it keeps refusing on an origin whose certificate is untrusted, even after you tap
through Safari's warning.** In that state `navigator.mediaDevices` is simply absent, so
you never even see a permission prompt. The capture page detects exactly this and says
so.

So pick one:

**A. Trust the local CA** (stays on your network)

1. Open `http://<ip>:8770/ca.crt` on the phone → allow the profile download.
2. *Settings → General → VPN & Device Management* → install it.
3. *Settings → General → About → **Certificate Trust Settings*** → switch it on.

Step 3 is the one that matters; installing without it leaves the cert untrusted.

**B. Tunnel one port** for a genuinely trusted certificate, no install:

```bash
cloudflared tunnel --url http://localhost:8770
```

Open the `https://…trycloudflare.com/capture.html` URL it prints. Pages and the
WebSocket share a single port precisely so this works. Trade-off: **your camera frames
transit Cloudflare**, so prefer A if that matters to you.

Safari gives web pages orientation but **not position**, so the cloud rotates with the
phone but cannot be walked around — fusing a real scan needs the native app's 6DoF pose.

## What's here

| Path | |
|---|---|
| `server/lidar_server.py` | The bridge. All ingest and fan-out. |
| `server/depth_model.py` | Monocular metric depth on the GPU, for anything with no scanner. |
| `server/tls.py` | Local CA + leaf cert, because iOS needs HTTPS to grant camera. |
| `server/protocol.py` | Wire format, shared by every Python piece. |
| `server/synth_phone.py` | Fake phone: raytraces a room. `--rgb-only` emulates a scannerless device. |
| `server/test_wire_compat.py` | Emulates the Swift client byte-for-byte and checks the round trip. |
| `server/test_web_capture.py` | Drives the browser capture path over WSS and checks the result. |
| `web/index.html` | The viewer. Raw WebGL2, no dependencies, no CDN. |
| `web/capture.html` | Phone capture page: camera + orientation over WSS. |
| `ios/` | The iPhone app (Swift + ARKit) and its XcodeGen spec. |
| `.github/workflows/build-ipa.yml` | Builds an unsigned IPA on a macOS runner. |

### Ports

| | plain | TLS |
|---|---|---|
| pages + websocket | 7770 | 8443 |
| native app ingest (raw TCP) | 7771 | — |

## Run the desktop side

```bash
cd server && python lidar_server.py
```

Open <http://localhost:8770>. The page tells you which address to type into the phone.

To drive it with no hardware at all — `--rgb-only` exercises the GPU depth path:

```bash
cd server && python synth_phone.py --rgb-only
```

Useful flags: `--no-depth-model` (skip loading the model entirely, LiDAR phones only),
`--depth-input 518`, `--depth-out 384`, `--depth-scale 1.0`.

### Display controls

`drag` orbit · `shift+drag` pan · `wheel` zoom · `R` reset · `B` build scan ·
`F` follow the phone · `1`–`4` colour mode (camera / depth / confidence / normal).

**Build scan** accumulates frames into one world-space cloud using the phone's pose,
so walking around a room fuses into a single model — then **Export .ply**.

**depth scale** appears only for inferred depth. Absolute scale from a single image is
ambiguous, so point the phone at something you can measure and slide until the size is
right. Clear the scan after changing it, or you will mix two scales.

## Get the app onto the iPhone

Xcode does not run on Windows, so a macOS CI runner compiles it and your own
`fuckapple` sideloader signs and installs it.

**1. Open the firewall for the bridge.** This is the step that will otherwise waste
your afternoon: as far as the phone is concerned nothing is listening, and the app just
sits on "waiting". Your LAN adapters are on the *Private* profile and no rule covers
these ports. In an **admin** PowerShell:

```powershell
New-NetFirewallRule -DisplayName "iPhone LiDAR bridge" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8770-8772 -Profile Private
```

A port rule rather than a program rule on purpose — the interpreter path changes
depending on which Python launches the bridge.

**2. Build the IPA.** The `iphone-lidar` branch of `itsTurdle/downloads` carries this
tree; pushing it runs `build-ipa` on a `macos-15` runner. Download the
`LidarStream-ipa` artifact from the run and unzip it to get `LidarStream.ipa`.

**3. Sideload it** with `fuckapple` (Documents/VS Code/fuckapple) — your Apple ID is
already saved and the signing cert is valid to 2027-01-15.

**4. First launch.** Allow **Camera** and — separately — **Local Network**. iOS will
not let the app reach your PC without that second one, and it only prompts once. Type
the address the display page shows, tap Connect.

> A free Apple ID signs for 7 days; `fuckapple` tracks this and reinstalls every 6.

## Device support

| | depth | pose | fused scan |
|---|---|---|---|
| iPhone 12 Pro and later, iPad Pro | LiDAR sensor | 6DoF | yes |
| **iPhone Air (`iPhone18,4`) — this phone** | **inferred on the GPU** | **6DoF** | **yes** |
| any other ARKit iPhone | inferred on the GPU | 6DoF | yes |

LiDAR is Pro-only; the Air has a single rear camera, so it can produce neither stereo
disparity nor live hardware depth (its Portrait depth is photo-only). World tracking
works on every ARKit device though, so the pose is always real — only the depth is
estimated, and it is estimated here rather than on the phone.

Inferred depth has no confidence channel, so the confidence filter and colour mode are
inert on that path.

## Requirements

- The ML path needs `torch` with CUDA plus `transformers`. Measured on the RTX PRO 6000
  Blackwell: **15.2 ms end to end per frame (~65 fps)**, of which ~11 ms is the model.
  Without CUDA the bridge still starts and says so, but scannerless phones then have no
  way to become depth.
- Roughly 60–200 KiB/s on the wire in RGB mode, 1.0–1.5 MiB/s in LiDAR mode.
- Keep the app in the foreground. iOS suspends ARKit in the background.

## Verification status

Verified end to end against the synthetic phone:

- **Geometry is correct.** Accumulating 1.26 M LiDAR-path points over 14 s of camera
  motion reconstructed a single coherent room whose bounding box matched the
  ground-truth geometry to within sensor noise (far wall −5.09 m vs −5.00 m actual).
  A wrong pose or axis convention would have ghosted instead.
- **Wire format is compatible.** `test_wire_compat.py` hand-builds frames exactly as
  `FrameStreamer.swift` does and passes 14/14 checks.
- **GPU depth path works.** `--rgb-only` produces `src=rgb+ml` at 384×288 / 110 k
  points, and moving pre/post-processing onto the GPU cut latency 32.5 → 15.2 ms.
- **Browser capture works.** `test_web_capture.py` passes 10/10, including verifying
  the certificate against the local CA (the same chain the phone checks) and that
  intrinsics are rescaled from the JPEG grid to the depth grid. The page itself was
  driven headlessly against a synthetic camera: 61 frames at 10.7 fps, camera acquired
  at 1280×960, no errors.
- **Both pages fit a phone.** Checked at a 390×844 viewport with device-metrics
  emulation: no horizontal overflow, HUD and control sheet do not overlap.
- **Reachable from another machine.** Verified from the Pi at 192.168.50.13 — ports
  8770/8771/8772 open, while a deliberately uncovered port with a live listener on it
  stayed blocked, which is what makes the firewall rule the proven cause.

The Swift itself is **compiled for the first time by CI** — there is no Xcode on
Windows. Expect to fix a compile error or two on the first run; the workflow prints the
full build log.
