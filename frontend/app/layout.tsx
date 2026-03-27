import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "AI Plagiarism Detector",
  description: "Sentence-level plagiarism checker using TF-IDF and BERT similarity",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
