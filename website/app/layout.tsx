import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  metadataBase: new URL("https://phonon.sh"),
  title: "Phonon — Open-source voice typing for Mac",
  description:
    "Fast, local, simple voice typing that understands your vocabulary and runs on your Mac.",
  icons: {
    icon: "/phonon-icon.png",
    apple: "/phonon-icon.png",
  },
  openGraph: {
    title: "Phonon — Fast. Local. Simple.",
    description: "Open-source voice typing for Mac.",
    type: "website",
    url: "https://phonon.sh",
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className={`${geistSans.variable} ${geistMono.variable}`}>{children}</body>
    </html>
  );
}
