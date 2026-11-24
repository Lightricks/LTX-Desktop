import { Text } from "@ds/Text/Text";
import { useLayoutEffect, useState } from "react";

import { VideoPlayer } from "../../../components/VideoView/VideoPlayer";

import styles from "./ExampleEmpty.module.scss";
import { type ExampleEmptyPlayback, exampleEmptyPlayback } from "./exampleMedia";

export function ExampleEmptyMedia({
  posterUrl,
  videoUrl,
}: {
  posterUrl: string;
  videoUrl: string;
}) {
  const [playback, setPlayback] = useState<ExampleEmptyPlayback>("poster");

  useLayoutEffect(() => {
    setPlayback(
      exampleEmptyPlayback(videoUrl, (mime) =>
        document.createElement("video").canPlayType(mime),
      ),
    );
  }, [videoUrl]);

  return (
    <div className={styles.root} data-testid="output-result-example-empty">
      <Text as="h3" variant="heading" size="sm" align="center" className={styles.title}>
        Example
      </Text>
      <div className={styles.frame}>
        {playback === "video" ? (
          <VideoPlayer
            src={videoUrl}
            posterUrl={posterUrl}
            objectFit="cover"
            label="Example video"
          />
        ) : (
          <img src={posterUrl} alt="" aria-hidden className={styles.media} />
        )}
      </div>
    </div>
  );
}
