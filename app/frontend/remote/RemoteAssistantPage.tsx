import { HistoryNavButtons } from "@/components/home/HistoryNavButtons";
import { DesktopAssistantPanel } from "@/ltx-io/components/Assistant/DesktopAssistantPanel";
import { GenerationQueuePanel } from "@/ltx-io/components/GenerationQueue/GenerationQueuePanel";
import { useExploreRuntime } from "@/ltx-io/runtime/ExploreRuntime";

export function RemoteAssistantPage() {
  const { api, generationPolling, mediaUrlForAsset, thumbUrlForAsset } = useExploreRuntime();
  return (
    <DesktopAssistantPanel
      open
      fullPage
      headerLeading={<HistoryNavButtons />}
      headerActions={
        <GenerationQueuePanel
          api={api}
          generationPolling={generationPolling}
          mediaUrlForAsset={mediaUrlForAsset}
          thumbUrlForAsset={thumbUrlForAsset}
          placement="header"
        />
      }
    />
  );
}
