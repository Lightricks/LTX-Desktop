import React from "react";

// List of DropdownItems components that accept selection state props
const supportedComponents = ["ItemAction", "ItemActionInformative", "ItemSound"];

type RenderSupportedOptionWithAdditionalPropsProps = {
  optionRender: React.ReactNode;
  isSelected: boolean;
  isDisabled?: boolean;
  isSelectable?: boolean;
};

/**
 * Renders an option's render prop, conditionally applying isDisabled and isSelected props
 * only to DropdownItems components that support these props.
 */
export function renderSupportedOptionWithAdditionalProps({
  optionRender,
  isDisabled,
  isSelected,
  isSelectable,
}: RenderSupportedOptionWithAdditionalPropsProps): React.ReactNode {
  if (!React.isValidElement(optionRender)) {
    return optionRender;
  }

  // Get component name safely
  const componentType = optionRender.type;
  let componentName = "";

  if (typeof componentType === "function" || typeof componentType === "object") {
    componentName =
      (componentType as { displayName?: string; name?: string }).displayName ||
      (componentType as { displayName?: string; name?: string }).name ||
      "";
  } else if (typeof componentType === "string") {
    componentName = componentType;
  }

  const isSupported = supportedComponents.some((name) =>
    componentName.toLowerCase().includes(name.toLowerCase()),
  );

  if (isSupported) {
    return React.cloneElement(
      optionRender as React.ReactElement<{
        isDisabled?: boolean;
        isSelected?: boolean;
        isSelectable?: boolean;
      }>,
      {
        isDisabled,
        isSelected,
        isSelectable,
      },
    );
  }

  return optionRender;
}
