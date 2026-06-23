import {
  Wand2,
  ShieldCheck,
  SlidersHorizontal,
  Gauge,
  Code2,
  BarChart3,
} from "lucide-react";
import Reveal from "@/components/ui/Reveal";

const FEATURES = [
  {
    icon: Wand2,
    title: "Deep humanization",
    desc: "Restructures phrasing, perplexity and rhythm so output reads like a real person wrote it — not a model.",
    glow: "shadow-glow",
  },
  {
    icon: ShieldCheck,
    title: "Detector-resistant",
    desc: "Engineered to lower AI-probability scores while preserving your original meaning and intent.",
    glow: "shadow-glow-violet",
  },
  {
    icon: SlidersHorizontal,
    title: "Tone control",
    desc: "Switch between Standard, Professional, Casual and Academic voices to match any assignment or audience.",
    glow: "shadow-glow-cyan",
  },
  {
    icon: Gauge,
    title: "Instant results",
    desc: "Async FastAPI backend returns humanized text in milliseconds, with a clear before/after diff view.",
    glow: "shadow-glow",
  },
  {
    icon: Code2,
    title: "Developer API",
    desc: "Generate API keys and integrate humanization into your own apps with a single REST endpoint.",
    glow: "shadow-glow-violet",
  },
  {
    icon: BarChart3,
    title: "Usage analytics",
    desc: "Track requests, latency and success rates from a real-time dashboard built for power users.",
    glow: "shadow-glow-cyan",
  },
];

export default function Features() {
  return (
    <section id="features" className="section-pad relative">
      <div className="container-x">
        <Reveal className="mx-auto max-w-2xl text-center">
          <span className="chip">Features</span>
          <h2 className="mt-4 font-display text-3xl font-bold sm:text-5xl">
            Everything you need to{" "}
            <span className="gradient-text">sound human</span>
          </h2>
          <p className="mt-4 text-slate-400">
            A complete toolkit — from one-click rewriting to a production-grade
            API, wrapped in a beautiful, fast interface.
          </p>
        </Reveal>

        <div className="mt-14 grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
          {FEATURES.map((f, i) => (
            <Reveal key={f.title} delay={(i % 3) * 0.08}>
              <article className="group glass h-full rounded-2xl p-6 transition duration-300 hover:-translate-y-1 hover:border-white/20">
                <div
                  className={`grid h-12 w-12 place-items-center rounded-xl bg-brand-gradient ${f.glow}`}
                >
                  <f.icon className="h-6 w-6 text-white" />
                </div>
                <h3 className="mt-5 font-display text-lg font-semibold">
                  {f.title}
                </h3>
                <p className="mt-2 text-sm leading-relaxed text-slate-400">
                  {f.desc}
                </p>
              </article>
            </Reveal>
          ))}
        </div>
      </div>
    </section>
  );
}
