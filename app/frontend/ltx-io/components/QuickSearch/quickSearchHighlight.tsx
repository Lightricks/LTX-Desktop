import { tokenizeQuickSearchQuery } from "./quickSearchModel.ts";

export function HighlightedQuery({
  text,
  query,
}: {
  text: string;
  query: string;
}) {
  const tokens = tokenizeQuickSearchQuery(query);
  if (tokens.length === 0) return text;

  const pattern = new RegExp(
    `(${tokens.map((token) => token.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})`,
    "gi",
  );
  const parts = text.split(pattern);
  return (
    <>
      {parts.map((part, index) =>
        tokens.some((token) => part.toLowerCase() === token) ? (
          <mark key={`${part}-${index}`}>{part}</mark>
        ) : (
          part
        ),
      )}
    </>
  );
}
