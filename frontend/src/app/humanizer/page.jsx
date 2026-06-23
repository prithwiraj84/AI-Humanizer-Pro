import Protected from "@/components/auth/Protected";
import HumanizerTool from "@/components/tool/HumanizerTool";

export const metadata = {
  title: "Humanizer Tool",
  description:
    "Paste AI-generated text and instantly humanize it with adjustable tone and deep mode.",
  robots: { index: false, follow: false },
};

export default function HumanizerPage() {
  return (
    <Protected>
      <HumanizerTool />
    </Protected>
  );
}
