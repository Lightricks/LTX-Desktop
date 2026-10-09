import { useEffect, useRef, useState } from "react";

import { Badge } from "@ds/Badge/Badge";
import { Button } from "@ds/Button/Button";
import { Flex } from "@ds/layout/Flex/Flex";
import { Text } from "@ds/Text/Text";
import { Check, Copy } from "lucide-react";

import { copyTextToClipboard } from "@/lib/copy-to-clipboard";
import { formatRelativeTime } from "../../../lib/formatRelativeTime";
import {
  type Generation,
} from "../../../lib/resultsFeedModel";
import { generationPresentation } from "../../../lib/generationPresentation.ts";
import styles from "./ResultMetaBar.module.scss";

const COPIED_FEEDBACK_MS = 2000;

function CopyPromptButton({ prompt }: { prompt: string }) {
  const [copied, setCopied] = useState(false);
  const timerRef = useRef<number | undefined>(undefined);
  const mountedRef = useRef(true);
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      window.clearTimeout(timerRef.current);
    };
  }, []);

  const label = copied ? "Prompt copied" : "Copy prompt";
  return (
    <Button
      appearance="overlay"
      hierarchy="secondary"
      size="md"
      isIconOnly
      className={styles.copyButton}
      leftIcon={copied ? <Check /> : <Copy />}
      aria-label={label}
      title={label}
      onClick={() => {
        void copyTextToClipboard(prompt).then((ok) => {
          if (!ok || !mountedRef.current) return;
          setCopied(true);
          window.clearTimeout(timerRef.current);
          timerRef.current = window.setTimeout(
            () => setCopied(false),
            COPIED_FEEDBACK_MS,
          );
        });
      }}
    />
  );
}

export function ResultMetaBar({ generation }: { generation: Generation }) {
  const { prompt, badges } = generationPresentation(generation);
  const timeAgo = formatRelativeTime(generation.created_at);

  if (!prompt && badges.length === 0 && !timeAgo) {
    return null;
  }

  return (
    <Flex
      direction="row"
      gap="lg"
      align="center"
      justify="between"
      className={styles.container}
    >
      <Flex direction="row" gap="md" align="center" className={styles.left}>
        <Flex direction="row" gap="md" align="center" className={styles.promptGroup}>
          {prompt ? (
            <Text
              as="span"
              variant="body"
              size="md"
              shouldTruncate
              title={prompt}
              className={styles.prompt}
            >
              {prompt}
            </Text>
          ) : null}
          {prompt ? <CopyPromptButton prompt={prompt} /> : null}
          {badges.length > 0 ? (
            <Flex gap="xs" align="center" className={styles.badges}>
              {badges.map((badge) => (
                <Badge
                  key={badge}
                  appearance="overlay"
                  className={styles.badge}
                  textVariant="body"
                  size="md"
                  text={badge}
                />
              ))}
            </Flex>
          ) : null}
        </Flex>
      </Flex>
      <Flex direction="row" gap="md" align="center" className={styles.trailing}>
        <Text
          as="span"
          size="md"
          variant="body"
          className={styles.timeAgo}
          shouldTruncate
        >
          {timeAgo}
        </Text>
      </Flex>
    </Flex>
  );
}
