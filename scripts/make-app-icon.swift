import AppKit

let destination = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
let variants = [
    ("icon_16x16.png", 16),
    ("icon_16x16@2x.png", 32),
    ("icon_32x32.png", 32),
    ("icon_32x32@2x.png", 64),
    ("icon_128x128.png", 128),
    ("icon_128x128@2x.png", 256),
    ("icon_256x256.png", 256),
    ("icon_256x256@2x.png", 512),
    ("icon_512x512.png", 512),
    ("icon_512x512@2x.png", 1024),
]

func render(size: Int, to url: URL) throws {
    let dimension = CGFloat(size)
    let image = NSImage(size: NSSize(width: dimension, height: dimension))
    image.lockFocus()

    let bounds = NSRect(x: 0, y: 0, width: dimension, height: dimension)
    let inset = dimension * 0.07
    let tile = NSBezierPath(
        roundedRect: bounds.insetBy(dx: inset, dy: inset),
        xRadius: dimension * 0.23,
        yRadius: dimension * 0.23
    )
    NSColor(calibratedRed: 0.067, green: 0.047, blue: 0.039, alpha: 1).setFill()
    tile.fill()

    let glowRect = NSRect(
        x: dimension * 0.16,
        y: dimension * 0.13,
        width: dimension * 0.68,
        height: dimension * 0.68
    )
    let glow = NSGradient(colors: [
        NSColor(calibratedRed: 1.0, green: 0.38, blue: 0.11, alpha: 0.92),
        NSColor(calibratedRed: 0.65, green: 0.12, blue: 0.025, alpha: 0.16),
        NSColor.clear,
    ])!
    glow.draw(in: NSBezierPath(ovalIn: glowRect), angle: -90)

    let mark = NSBezierPath()
    mark.lineWidth = max(1.5, dimension * 0.075)
    mark.lineCapStyle = .round
    mark.move(to: NSPoint(x: dimension * 0.29, y: dimension * 0.49))
    mark.curve(
        to: NSPoint(x: dimension * 0.71, y: dimension * 0.49),
        controlPoint1: NSPoint(x: dimension * 0.39, y: dimension * 0.71),
        controlPoint2: NSPoint(x: dimension * 0.61, y: dimension * 0.27)
    )
    NSColor(calibratedRed: 1.0, green: 0.84, blue: 0.70, alpha: 1).setStroke()
    mark.stroke()

    image.unlockFocus()
    guard let tiff = image.tiffRepresentation,
          let bitmap = NSBitmapImageRep(data: tiff),
          let png = bitmap.representation(using: .png, properties: [:])
    else { throw CocoaError(.fileWriteUnknown) }
    try png.write(to: url)
}

for (name, size) in variants {
    try render(size: size, to: destination.appendingPathComponent(name))
}
