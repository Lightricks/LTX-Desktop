export const useKeyboardHandlers = () => {
  const handlePromptKeyDown =
    (onKeyPress: () => void | Promise<void>) =>
    (event: React.KeyboardEvent<HTMLTextAreaElement | HTMLInputElement>) => {
      if (event.key === "Escape") {
        event.currentTarget.blur();
        return;
      }

      if (event.key === "Enter") {
        if (event.shiftKey) {
          return;
        }

        event.preventDefault();

        void onKeyPress();
      }
    };

  const handleKeyDownEnter =
    (onKeyPress: () => void | Promise<void>) => (event: React.KeyboardEvent) => {
      if (event.key === "Enter") {
        event.preventDefault();
        void onKeyPress();
      }
    };

  /**
   * Keyboard handler for elements with role="button".
   * Triggers a click on Enter or Space, matching native button behavior.
   */
  const handleButtonKeyDown = (event: React.KeyboardEvent<HTMLElement>) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      event.currentTarget.click();
    }
  };

  return { handlePromptKeyDown, handleKeyDownEnter, handleButtonKeyDown };
};
