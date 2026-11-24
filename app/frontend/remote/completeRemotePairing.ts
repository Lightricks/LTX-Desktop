import { pairDevice } from "./api.ts";

export async function completeRemotePairing(
  grant: string | null,
  navigate: (to: string, options: { replace: boolean }) => void,
): Promise<"ok" | "missing" | "failed"> {
  const paired = await pairDevice(grant);
  if (paired === "ok") {
    navigate("/", { replace: true });
  }
  return paired;
}
