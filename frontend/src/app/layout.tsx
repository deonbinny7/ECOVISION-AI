import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "EcoVision AI - Explainable Waste Classification",
  description: "An industry-grade AI product combining cinematic storytelling and explainable AI to automate waste classification.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} min-h-screen bg-slate-900 text-slate-100 antialiased selection:bg-cyan-500/30`}>
        {children}
      </body>
    </html>
  );
}
