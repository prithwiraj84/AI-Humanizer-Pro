import SignupForm from "@/components/auth/SignupForm";

export const metadata = {
  title: "Sign up",
  robots: { index: false, follow: false },
};

export default function SignupPage() {
  return <SignupForm />;
}
