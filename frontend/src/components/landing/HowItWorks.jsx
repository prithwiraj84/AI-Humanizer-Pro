import { ClipboardPaste, SlidersHorizontal, Sparkles } from "lucide-react";
import Reveal from "@/components/ui/Reveal";

const STEPS = [
  {
    n: "01",
    icon: ClipboardPaste,
    title: "Paste your draft",
    desc: "Drop in any AI-generated text — an essay, report, email or article.",
  },
  {
    n: "02",
    icon: SlidersHorizontal,
    title: "Pick tone & mode",
    desc: "Choose a voice and toggle Deep Mode for maximum humanization.",
  },
  {
    n: "03",
    icon: Sparkles,
    title: "Get human text",
    desc: "Receive natural, detector-resistant writing with a full before/after diff.",
  },
];

export default function HowItWorks() {
  return (
    <section id="how" className="section-pad relative">
      <div className="container-x">
        <Reveal className="mx-auto max-w-2xl text-center">
          <span className="chip">How it works</span>
          <h2 className="mt-4 font-display text-3xl font-bold sm:text-5xl">
            Three steps to <span className="gradient-text">human</span>
          </h2>
        </Reveal>

        <div className="relative mt-14 grid gap-6 md:grid-cols-3">
          <div className="absolute left-0 right-0 top-12 hidden h-px bg-gradient-to-r from-transparent via-white/15 to-transparent md:block" />
          {STEPS.map((s, i) => (
            <Reveal key={s.n} delay={i * 0.12}>
              <div className="glass relative h-full rounded-2xl p-7 text-center">
                <div className="mx-auto grid h-14 w-14 place-items-center rounded-2xl bg-ink-800 ring-1 ring-white/10">
                  <s.icon className="h-6 w-6 text-brand-cyan" />
                </div>
                <div className="mt-4 font-mono text-xs tracking-widest text-brand-violet">
                  STEP {s.n}
                </div>
                <h3 className="mt-1 font-display text-lg font-semibold">
                  {s.title}
                </h3>
                <p className="mt-2 text-sm leading-relaxed text-slate-400">
                  {s.desc}
                </p>
              </div>
            </Reveal>
          ))}
        </div>
      </div>
    </section>
  );
}
