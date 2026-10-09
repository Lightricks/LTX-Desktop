export function featureDetailsHeroPlayerProps(
  hero: { videoUrl: string; posterUrl: string },
  title: string,
  reduceMotion: boolean | null,
): {
  src: string;
  posterUrl: string;
  objectFit: "cover";
  label: string;
  autoPlay: boolean;
} {
  return {
    src: hero.videoUrl,
    posterUrl: hero.posterUrl,
    objectFit: "cover",
    label: title,
    autoPlay: reduceMotion !== true,
  };
}
