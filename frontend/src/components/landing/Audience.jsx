import { GraduationCap, Briefcase, Check } from "lucide-react";
import Reveal from "@/components/ui/Reveal";

const AUDIENCES = [
  {
    icon: GraduationCap,
    tag: "For students",
    title: "Submit work that sounds like you",
    points: [
      "Humanize essays, theses & assignments",
      "Academic tone that fits any rubric",
      "Lower AI-detection flags with confidence",
    ],
    accent: "from-brand-indigo to-brand-violet",
  },
  {
    icon: Briefcase,
    tag: "For professionals",
    title: "Polished comms in seconds",
    points: [
      "Emails, reports & proposals that feel personal",
      "Professional & casual voices on demand",
      "API to automate your team's workflows",
    ],
    accent: "from-brand-cyan to-brand-indigo",
  },
];

export default function Audience() {
  return (
    <section className="section-pad relative">
      <div className="container-x">
        <Reveal className="mx-auto max-w-2xl text-center">
          <span className="chip">Made for you</span>
          <h2 className="mt-4 font-display text-3xl font-bold sm:text-5xl">
            Built for students &{" "}
            <span className="gradient-text">working professionals</span>
          </h2>
        </Reveal>

        <div className="mt-14 grid gap-5 md:grid-cols-2">
          {AUDIENCES.map((a, i) => (
            <Reveal key={a.tag} delay={i * 0.1}>
              <article className="glass h-full overflow-hidden rounded-2xl p-7">
                <div
                  className={`grid h-12 w-12 place-items-center rounded-xl bg-gradient-to-br ${a.accent} shadow-glow`}
                >
                  <a.icon className="h-6 w-6 text-white" />
                </div>
                <div className="mt-5 text-xs font-semibold uppercase tracking-widest text-brand-violet">
                  {a.tag}
                </div>
                <h3 className="mt-1 font-display text-xl font-semibold">
                  {a.title}
                </h3>
                <ul className="mt-5 space-y-3">
                  {a.points.map((p) => (
                    <li key={p} className="flex items-start gap-3 text-sm text-slate-300">
                      <span className="mt-0.5 grid h-5 w-5 shrink-0 place-items-center rounded-full bg-brand-emerald/15 text-brand-emerald">
                        <Check className="h-3.5 w-3.5" />
                      </span>
                      {p}
                    </li>
                  ))}
                </ul>
              </article>
            </Reveal>
          ))}
        </div>
      </div>
    </section>
  );
}
