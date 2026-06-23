import Link from "next/link";
import { Sparkles, Github, Twitter, Mail } from "lucide-react";
import { SITE } from "@/lib/site";

const cols = [
  {
    title: "Product",
    links: [
      { label: "Humanizer", href: "/humanizer" },
      { label: "Features", href: "/#features" },
      { label: "How it works", href: "/#how" },
      { label: "Dashboard", href: "/dashboard" },
    ],
  },
  {
    title: "Developers",
    links: [
      { label: "API Docs", href: "/docs" },
      { label: "Get API key", href: "/dashboard" },
      { label: "Status", href: "/#stats" },
    ],
  },
  {
    title: "Account",
    links: [
      { label: "Log in", href: "/login" },
      { label: "Sign up", href: "/signup" },
      { label: "Reset password", href: "/forgot-password" },
    ],
  },
];

export default function Footer() {
  return (
    <footer className="relative border-t border-white/10 bg-ink-950/60">
      <div className="container-x px-5 py-14 sm:px-8">
        <div className="grid gap-10 md:grid-cols-[1.4fr_1fr_1fr_1fr]">
          <div>
            <Link href="/" className="flex items-center gap-2 font-display text-lg font-bold">
              <span className="grid h-9 w-9 place-items-center rounded-xl bg-brand-gradient shadow-glow">
                <Sparkles className="h-5 w-5 text-white" />
              </span>
              <span>
                AI Humanizer<span className="gradient-text"> Pro</span>
              </span>
            </Link>
            <p className="mt-4 max-w-xs text-sm leading-relaxed text-slate-400">
              Transform AI-generated drafts into natural, human writing — built
              for students and working professionals.
            </p>
            <div className="mt-5 flex gap-3">
              {[
                { Icon: Github, label: "GitHub", href: "https://github.com" },
                { Icon: Twitter, label: "Twitter", href: "https://twitter.com" },
                { Icon: Mail, label: "Email us", href: "mailto:hello@aihumanizer.pro" },
              ].map(({ Icon, label, href }) => (
                <a
                  key={label}
                  href={href}
                  aria-label={label}
                  target={href.startsWith("http") ? "_blank" : undefined}
                  rel={href.startsWith("http") ? "noopener noreferrer" : undefined}
                  className="grid h-9 w-9 place-items-center rounded-lg border border-white/10 bg-white/5 text-slate-300 transition hover:border-white/25 hover:text-white"
                >
                  <Icon className="h-4 w-4" />
                </a>
              ))}
            </div>
          </div>

          {cols.map((col) => (
            <div key={col.title}>
              <h4 className="font-display text-sm font-semibold text-white">
                {col.title}
              </h4>
              <ul className="mt-4 space-y-2.5">
                {col.links.map((l) => (
                  <li key={l.label}>
                    <Link
                      href={l.href}
                      className="text-sm text-slate-400 transition hover:text-white"
                    >
                      {l.label}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        <div className="mt-12 flex flex-col items-center justify-between gap-3 border-t border-white/10 pt-6 text-xs text-slate-400 sm:flex-row">
          <p>
            © {new Date().getFullYear()} {SITE.name}. All rights reserved.
          </p>
          <p>Crafted with Next.js, Three.js & FastAPI.</p>
        </div>
      </div>
    </footer>
  );
}
