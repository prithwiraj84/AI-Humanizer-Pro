// Central site metadata — single source of truth for SEO.
export const SITE = {
  name: "AI Humanizer Pro",
  shortName: "AI Humanizer",
  url: process.env.NEXT_PUBLIC_SITE_URL || "http://localhost:3000",
  description:
    "AI Humanizer Pro rewrites AI-generated text into natural, human-sounding writing that bypasses AI detectors. Built for students and working professionals — adjustable tone, deep mode, and a developer API.",
  keywords: [
    "AI humanizer",
    "humanize AI text",
    "AI detector bypass",
    "undetectable AI",
    "paraphrasing tool",
    "AI to human text",
    "ChatGPT humanizer",
    "essay humanizer",
    "AI content rewriter",
  ],
  twitter: "@aihumanizer",
  author: "AI Humanizer Pro",
};

export const NAV_LINKS = [
  { label: "Features", href: "/#features" },
  { label: "How it works", href: "/#how" },
  { label: "Humanizer", href: "/humanizer" },
  { label: "Docs", href: "/docs" },
];
