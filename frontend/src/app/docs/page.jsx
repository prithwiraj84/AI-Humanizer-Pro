import Navbar from "@/components/layout/Navbar";
import Footer from "@/components/layout/Footer";
import CodeBlock from "@/components/docs/CodeBlock";
import { KeyRound, Terminal, Braces } from "lucide-react";

export const metadata = {
  title: "API Documentation",
  description:
    "Integrate AI Humanizer Pro into your app with a single REST endpoint. Authenticate with an API key and POST text to /humanize.",
  alternates: { canonical: "/docs" },
};

const REQUEST_PARAMS = [
  { name: "text", type: "string", req: "Required", desc: "The AI-generated text to humanize." },
  { name: "tone", type: "string", req: "Optional", desc: "standard · professional · casual · academic. Defaults to standard." },
  { name: "deep_mode", type: "boolean", req: "Optional", desc: "Enables maximum humanization. Defaults to false." },
];

const RESPONSE_FIELDS = [
  { name: "humanized", type: "string", desc: "The rewritten, human-sounding text." },
  { name: "ai_probability", type: "string", desc: "Estimated AI-detection probability, e.g. \"4.0%\"." },
  { name: "confidence", type: "string", desc: "Confidence of the human classification." },
  { name: "classification", type: "string", desc: "Human-Written." },
  { name: "structural_changes", type: "string", desc: "How much sentence structure changed." },
  { name: "changed_words", type: "number", desc: "Count of words modified." },
  { name: "longest_unchanged", type: "string", desc: "Longest phrase left untouched." },
  { name: "diff_html", type: "string", desc: "HTML diff with added/unchanged spans." },
];

const curl = `curl -X POST https://noai.devprithwiraj.in/humanize \\
  -H "Content-Type: application/json" \\
  -H "X-API-Key: hmnz_your_api_key_here" \\
  -d '{
    "text": "Your AI generated text here",
    "tone": "professional",
    "deep_mode": true
  }'`;

const js = `const res = await fetch("https://noai.devprithwiraj.in/humanize", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "X-API-Key": "hmnz_your_api_key_here",
  },
  body: JSON.stringify({
    text: "Your AI generated text here",
    tone: "professional",
    deep_mode: true,
  }),
});

const data = await res.json();
console.log(data.humanized);`;

const py = `import requests

res = requests.post(
    "https://noai.devprithwiraj.in/humanize",
    headers={
        "Content-Type": "application/json",
        "X-API-Key": "hmnz_your_api_key_here",
    },
    json={
        "text": "Your AI generated text here",
        "tone": "professional",
        "deep_mode": True,
    },
)

print(res.json()["humanized"])`;

export default function DocsPage() {
  return (
    <>
      <Navbar />
      <main className="container-x px-5 pb-24 pt-28 sm:px-8">
        <header className="max-w-2xl">
          <span className="chip">Developers</span>
          <h1 className="mt-4 font-display text-4xl font-bold sm:text-5xl">
            API <span className="gradient-text">Documentation</span>
          </h1>
          <p className="mt-4 text-slate-400">
            Humanize text programmatically with a single REST endpoint.
            Authenticate with an API key from your{" "}
            <a href="/dashboard" className="text-indigo-300 hover:underline">
              dashboard
            </a>
            .
          </p>
        </header>

        <div className="mt-12 grid gap-10">
          {/* Auth */}
          <section className="glass rounded-2xl p-7">
            <div className="flex items-center gap-2">
              <KeyRound className="h-5 w-5 text-brand-violet" />
              <h2 className="font-display text-xl font-semibold">Authentication</h2>
            </div>
            <p className="mt-3 text-sm text-slate-400">
              Pass your secret key in the <code className="rounded bg-white/10 px-1.5 py-0.5 font-mono text-xs">X-API-Key</code> header on every request.
            </p>
            <div className="mt-4">
              <CodeBlock lang="header" code={`X-API-Key: hmnz_your_api_key_here`} />
            </div>
          </section>

          {/* Endpoint */}
          <section className="glass rounded-2xl p-7">
            <div className="flex items-center gap-2">
              <Terminal className="h-5 w-5 text-brand-cyan" />
              <h2 className="font-display text-xl font-semibold">Endpoint</h2>
            </div>
            <div className="mt-4 inline-flex items-center gap-3 rounded-lg border border-white/10 bg-ink-900/60 px-4 py-2 font-mono text-sm">
              <span className="rounded bg-brand-emerald/15 px-2 py-0.5 text-xs font-bold text-brand-emerald">
                POST
              </span>
              <span className="text-slate-200">/humanize</span>
            </div>

            <h3 className="mt-7 font-display text-sm font-semibold uppercase tracking-wider text-slate-400">
              Request body
            </h3>
            <div className="mt-3 overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead className="text-xs uppercase tracking-wider text-slate-500">
                  <tr>
                    <th className="pb-3 pr-4 font-medium">Field</th>
                    <th className="pb-3 pr-4 font-medium">Type</th>
                    <th className="pb-3 pr-4 font-medium">Required</th>
                    <th className="pb-3 font-medium">Description</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5">
                  {REQUEST_PARAMS.map((p) => (
                    <tr key={p.name}>
                      <td className="py-3 pr-4 font-mono text-xs text-brand-cyan">{p.name}</td>
                      <td className="py-3 pr-4 text-slate-400">{p.type}</td>
                      <td className="py-3 pr-4 text-slate-400">{p.req}</td>
                      <td className="py-3 text-slate-300">{p.desc}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          {/* Examples */}
          <section className="glass rounded-2xl p-7">
            <div className="flex items-center gap-2">
              <Braces className="h-5 w-5 text-indigo-300" />
              <h2 className="font-display text-xl font-semibold">Examples</h2>
            </div>
            <div className="mt-5 grid gap-5">
              <CodeBlock lang="cURL" code={curl} />
              <CodeBlock lang="JavaScript" code={js} />
              <CodeBlock lang="Python" code={py} />
            </div>
          </section>

          {/* Response */}
          <section className="glass rounded-2xl p-7">
            <h2 className="font-display text-xl font-semibold">Response</h2>
            <div className="mt-4 overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead className="text-xs uppercase tracking-wider text-slate-500">
                  <tr>
                    <th className="pb-3 pr-4 font-medium">Field</th>
                    <th className="pb-3 pr-4 font-medium">Type</th>
                    <th className="pb-3 font-medium">Description</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5">
                  {RESPONSE_FIELDS.map((f) => (
                    <tr key={f.name}>
                      <td className="py-3 pr-4 font-mono text-xs text-brand-cyan">{f.name}</td>
                      <td className="py-3 pr-4 text-slate-400">{f.type}</td>
                      <td className="py-3 text-slate-300">{f.desc}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </div>
      </main>
      <Footer />
    </>
  );
}
