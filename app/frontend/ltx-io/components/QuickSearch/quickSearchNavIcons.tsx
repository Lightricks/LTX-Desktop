import type { ReactNode } from "react";
import { Radio } from "lucide-react";

import ProjectsIcon from "@/ds/assets/Icons/Projects.svg?react";
import AssetsIcon from "@/ds/assets/Icons/SectionVideos.svg?react";
import HomeIcon from "@/assets/home/icons/Home/Line.svg?react";

export function quickSearchIconForPageId(id: string): ReactNode {
  switch (id) {
    case "page-home":
      return <HomeIcon aria-hidden />;
    case "page-assets":
      return <AssetsIcon aria-hidden />;
    case "page-projects":
      return <ProjectsIcon aria-hidden />;
    default:
      return <Radio className="h-3.5 w-3.5" aria-hidden />;
  }
}
