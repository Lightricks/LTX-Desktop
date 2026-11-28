// localStorage wrapper that routes failures through the design system's error logger.
import { handleErrors } from "./handleErrors";

export class storage {
  static setItem(key: string, value: string): void {
    try {
      localStorage.setItem(key, value);
    } catch (error) {
      handleErrors([error]);
    }
  }

  static getItem(key: string): string | null {
    return localStorage.getItem(key);
  }

  static removeItem(key: string): void {
    localStorage.removeItem(key);
  }
}
