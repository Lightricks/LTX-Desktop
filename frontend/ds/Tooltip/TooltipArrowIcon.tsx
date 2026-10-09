import { type SVGProps } from "react";

type TooltipArrowIconProps = SVGProps<SVGSVGElement>;

export function TooltipArrowIcon(props: TooltipArrowIconProps) {
  return (
    <svg width={16} height={6} viewBox="0 0 16 6" {...props}>
      <path d="M0 6s1.796-.013 4.67-3.615C5.851.9 6.93.006 8 0c1.07-.006 2.148.887 3.343 2.385C14.233 6.005 16 6 16 6H0z" />
    </svg>
  );
}
