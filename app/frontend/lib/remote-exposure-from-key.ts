/** Next Off/On value for the Remote radiogroup. Arrows wrap; Home/End are absolute. */
export function remoteExposureFromKey(on: boolean, key: string): boolean | null {
  switch (key) {
    case "Home":
      return false;
    case "End":
      return true;
    case "ArrowRight":
    case "ArrowDown":
    case "ArrowLeft":
    case "ArrowUp":
      return !on;
    default:
      return null;
  }
}
