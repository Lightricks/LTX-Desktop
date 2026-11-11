import { IMAGE_BLOCKED, IMAGE_BLOCKED_MESSAGE } from "../../lib/staticFailureNotices";
import type { Generation } from "../../lib/resultsFeedModel";

const GENERIC_FAILURE = "The image could not be generated. Try again.";

export function imageFailureMessage(generation: Pick<Generation, "error_code">): string {
  return generation.error_code === IMAGE_BLOCKED ? IMAGE_BLOCKED_MESSAGE : GENERIC_FAILURE;
}
