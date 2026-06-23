import LoginForm from "@/components/auth/LoginForm";

export const metadata = {
  title: "Log in",
  robots: { index: false, follow: false },
};

export default function LoginPage() {
  return <LoginForm />;
}
