export function nextGenerateAfterDismiss(
  listed: { id: string; downloaded: boolean }[] | null,
  catalogId: string,
): "proceed" | "download" {
  const entry = listed?.find((item) => item.id === catalogId);
  return entry?.downloaded ? "proceed" : "download";
}
