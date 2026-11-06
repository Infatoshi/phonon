import AppKit
// Phonon icon: Okabe-Ito blue field, five white rounded bars of a voice level.
let size = 1024.0
let img = NSImage(size: NSSize(width: size, height: size))
img.lockFocus()
NSColor(red: 0.0, green: 0.45, blue: 0.70, alpha: 1).setFill()
NSRect(x: 0, y: 0, width: size, height: size).fill()
NSColor.white.setFill()
let heights = [0.22, 0.42, 0.62, 0.42, 0.22]
let barW = 92.0, gap = 58.0
let total = Double(heights.count) * barW + Double(heights.count - 1) * gap
for (i, h) in heights.enumerated() {
    let x = (size - total) / 2 + Double(i) * (barW + gap)
    let hh = h * size
    NSBezierPath(roundedRect: NSRect(x: x, y: (size - hh) / 2, width: barW, height: hh), xRadius: barW / 2, yRadius: barW / 2).fill()
}
img.unlockFocus()
let rep = NSBitmapImageRep(data: img.tiffRepresentation!)!
try! rep.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: CommandLine.arguments[1]))
