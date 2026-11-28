import { clsx } from "clsx";
import { Fragment, forwardRef } from "react";

import { Flex } from "@ds/layout/Flex/Flex";

import styles from "./SidePanelMenu.module.scss";
import { SidePanelMenuSection } from "./SidePanelMenuSection";

/**
 * Interface for the SidePanelMenu props
 * @interface SidePanelMenuProps
 */
export interface SidePanelMenuProps {
  /** Array of sections to display in the side panel menu */
  sections: SidePanelMenuSection[];
  /** Optional items to display in the footer */
  footer?: SidePanelMenuSection;
  /** Additional CSS class names to be applied to the container */
  className?: string;
}

/**
 * SidePanelMenu displays a vertical navigation menu organized into sections.
 * Each section can contain multiple menu items, and sections are separated by dividers.
 * Items are displayed as DynamicButtons.
 * @see {@link DynamicButton} for more information on the props.
 *
 * The component also includes a fixed Help button in the footer.
 *
 * @component
 * @example
 * ```tsx
 * // Basic usage
 * const sections = [
 *   {
 *     label: "Main Navigation",
 *     items: [
 *       { label: "Dashboard", icon: DashboardIcon, onClick: () => {} },
 *       { label: "Settings", icon: SettingsIcon, onClick: () => {} }
 *     ]
 *   }
 * ];
 *
 * const footer = [
 *   {
 *     label: "Help",
 *     icon: HelpIcon,
 *     onClick: () => {}
 *   }
 * ];
 *
 * <SidePanelMenu sections={sections} footer={footer} />
 *
 * // With custom className
 * <SidePanelMenu
 *   sections={sections}
 *   footer={footer}
 *   className="customClassName"
 * />
 * ```
 */
export const SidePanelMenu = forwardRef<HTMLDivElement, SidePanelMenuProps>(
  (props, ref): JSX.Element => {
    const { sections, className, footer } = props;

    return (
      <Flex ref={ref} direction="column" className={clsx(styles.container, className)}>
        <div className={styles.sections}>
          {sections.map((section, index) => (
            <Fragment key={index}>
              <SidePanelMenuSection
                label={section.label}
                labelVariant={section.labelVariant}
                labelSize={section.labelSize}
                items={section.items}
                rightSideElement={section.rightSideElement}
                shouldShowSeparator={index > 0}
                separatorClassName={section.separatorClassName}
                labelClassName={section.labelClassName}
              />
            </Fragment>
          ))}
        </div>
        {footer && (
          <div className={styles.footerButton}>
            <SidePanelMenuSection
              label={footer.label}
              labelVariant={footer.labelVariant}
              labelSize={footer.labelSize}
              items={footer.items}
              shouldShowSeparator={footer.shouldShowSeparator ?? true}
              separatorClassName={footer.separatorClassName}
              labelClassName={footer.labelClassName}
            />
          </div>
        )}
      </Flex>
    );
  },
);

SidePanelMenu.displayName = "SidePanelMenu";
