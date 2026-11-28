import { Button } from "@ds/Button/Button";
import { Stepper, type StepItem } from "@ds/Stepper/Stepper";
import { Flex } from "@ds/layout/Flex/Flex";
import { useKeyboardShortcut } from "@ds/lib/useKeyboardShortcut";
import { useCallback, useMemo, useRef, useState } from "react";

import { Modal } from "@/ltx-io/components/shared/Modal/Modal";

import {
  MOTION_PATH_MAX_TRACKS,
  type MotionPath,
  type MotionPathPoint,
} from "../../../lib/motionPathSampler";
import {
  MotionPathCanvas,
  type MotionPathCanvasTrack,
  type MotionPathTool,
} from "./MotionPathCanvas";
import styles from "./MotionPathEditModal.module.scss";
import { MotionPathPaceStep } from "./MotionPathPaceStep";
import { MotionPathToolbar } from "./MotionPathToolbar";
import { useMotionPathToolShortcuts } from "./useMotionPathToolShortcuts";
import { useTrackPaces } from "./useTrackPaces";

const MOTION_PATH_HISTORY_LIMIT = 50;

/** The model reads a dot by its place, so two close dots can trade their objects. */
const TRACKS_WARNING = "Tracks that meet can swap their objects. Keep them apart.";

const STEP_ITEMS: StepItem[] = [
  {
    title: "1. Draw a motion path",
    description: "Draw to guide an element's movement",
  },
  {
    title: "2. Set the pace",
    description: "Preview and tune timing along the path",
  },
];

/** A track. Its id stays the same through undo, and it keys the pace of the track. */
type Track = { id: number; points: MotionPathPoint[] };
/** The modal edits one track at a time. */
type Tracks = Track[];

export type MotionPathEditModalProps = {
  imageUrl: string;
  /** The frame the pipeline renders for the image, or null until the image loads. */
  imageAspect: number | null;
  /** The tracks to edit, or none to draw a new one. */
  paths: MotionPath[];
  durationSec: number;
  /** The clip lengths the form offers. */
  durationOptions: readonly number[];
  onAccept: (paths: MotionPath[], durationSec: number) => void;
  onClose: () => void;
};

/** Undo and redo, as in the other ltx.io editors: Cmd or Ctrl with Z, Shift for redo, or Y. */
function historyShortcut(event: KeyboardEvent): "undo" | "redo" | null {
  if (!(event.metaKey || event.ctrlKey) || event.altKey) return null;
  const key = event.key.toLowerCase();
  if (key === "y") return "redo";
  if (key !== "z") return null;
  return event.shiftKey ? "redo" : "undo";
}

/**
 * The dot color of a track. The clip colors the tracks that have a path in order, so an empty
 * track before it does not take a color.
 */
function colorIndexOf(tracks: Tracks, trackIndex: number): number {
  return tracks.slice(0, trackIndex).filter((track) => track.points.length > 0).length;
}

type TracksProblem = { message: string; trackIndex: number };

/** What stops the editor from going on, or null. An empty track is dropped, so it is fine. */
function findTracksProblem(tracks: Tracks): TracksProblem | null {
  const onePointIndex = tracks.findIndex((track) => track.points.length === 1);
  if (onePointIndex !== -1) {
    return { message: `Track ${onePointIndex + 1} needs at least 2 points`, trackIndex: onePointIndex };
  }
  if (tracks.every((track) => track.points.length === 0)) {
    return { message: "Draw a motion path", trackIndex: 0 };
  }
  return null;
}

export function MotionPathEditModal({
  imageUrl,
  imageAspect,
  paths,
  durationSec,
  durationOptions,
  onAccept,
  onClose,
}: MotionPathEditModalProps) {
  const [currentStep, setCurrentStep] = useState(0);
  const [draftTracks, setDraftTracks] = useState<Tracks>(() =>
    paths.length > 0
      ? paths.map((path, index) => ({ id: index, points: path.points }))
      : [{ id: 0, points: [] }],
  );
  const nextTrackIdRef = useRef(Math.max(paths.length, 1));
  const draftTracksRef = useRef(draftTracks);
  const [activeTrack, setActiveTrack] = useState(0);
  const [pastTracks, setPastTracks] = useState<Tracks[]>([]);
  const [futureTracks, setFutureTracks] = useState<Tracks[]>([]);
  const [selectedTool, setSelectedTool] = useState<MotionPathTool>("freehand");
  const { activeTool } = useMotionPathToolShortcuts({
    enabled: currentStep === 0,
    selectedTool,
    setSelectedTool,
  });
  const [acceptError, setAcceptError] = useState<string | null>(null);
  const pacing = useTrackPaces(paths, durationSec);

  const drawnTracks = useMemo(
    () => draftTracks.filter((track) => track.points.length > 0),
    [draftTracks],
  );
  const tracksProblem = useMemo(() => findTracksProblem(draftTracks), [draftTracks]);
  const canContinueToPace = tracksProblem === null;
  const activePoints = draftTracks[activeTrack]?.points ?? [];
  const otherTracks = useMemo<MotionPathCanvasTrack[]>(
    () =>
      draftTracks
        .map((track, index) => ({ points: track.points, colorIndex: colorIndexOf(draftTracks, index) }))
        .filter((_, index) => index !== activeTrack),
    [draftTracks, activeTrack],
  );

  const stepItems = useMemo<StepItem[]>(
    () =>
      STEP_ITEMS.map((item, index) => ({
        ...item,
        isDisabled: index === 1 ? !canContinueToPace : false,
      })),
    [canContinueToPace],
  );

  const updateDraftTracks = useCallback((tracks: Tracks) => {
    draftTracksRef.current = tracks;
    setDraftTracks(tracks);
    setActiveTrack((index) => Math.min(index, tracks.length - 1));
    if (findTracksProblem(tracks) === null) {
      setAcceptError(null);
    }
  }, []);

  // The updaters below run later, after `updateDraftTracks` replaced the ref. Each one must
  // append the snapshot taken here, not read the ref.
  const pushHistory = useCallback(() => {
    const snapshot = draftTracksRef.current;
    setPastTracks((history) => [...history, snapshot].slice(-MOTION_PATH_HISTORY_LIMIT));
    setFutureTracks([]);
  }, []);

  const handlePointsChange = useCallback(
    (points: MotionPathPoint[], historyMode: "push" | "replace" = "push") => {
      if (historyMode === "push") pushHistory();
      else setFutureTracks([]);
      updateDraftTracks(
        draftTracksRef.current.map((track, index) =>
          index === activeTrack ? { ...track, points } : track,
        ),
      );
    },
    [activeTrack, pushHistory, updateDraftTracks],
  );

  const handleUndo = useCallback(() => {
    const previous = pastTracks.at(-1);
    if (!previous) return;
    const snapshot = draftTracksRef.current;
    setPastTracks((history) => history.slice(0, -1));
    setFutureTracks((history) => [...history, snapshot].slice(-MOTION_PATH_HISTORY_LIMIT));
    updateDraftTracks(previous);
  }, [pastTracks, updateDraftTracks]);

  const handleRedo = useCallback(() => {
    const next = futureTracks.at(-1);
    if (!next) return;
    const snapshot = draftTracksRef.current;
    setFutureTracks((history) => history.slice(0, -1));
    setPastTracks((history) => [...history, snapshot].slice(-MOTION_PATH_HISTORY_LIMIT));
    updateDraftTracks(next);
  }, [futureTracks, updateDraftTracks]);

  const handleClear = useCallback(() => {
    if (activePoints.length === 0) return;
    handlePointsChange([]);
  }, [activePoints.length, handlePointsChange]);

  const handleAddTrack = useCallback(() => {
    if (draftTracksRef.current.length >= MOTION_PATH_MAX_TRACKS) return;
    pushHistory();
    updateDraftTracks([...draftTracksRef.current, { id: nextTrackIdRef.current++, points: [] }]);
    setActiveTrack(draftTracksRef.current.length - 1);
  }, [pushHistory, updateDraftTracks]);

  const handleRemoveTrack = useCallback(() => {
    if (draftTracksRef.current.length <= 1) return;
    pushHistory();
    updateDraftTracks(draftTracksRef.current.filter((_, index) => index !== activeTrack));
  }, [activeTrack, pushHistory, updateDraftTracks]);

  useKeyboardShortcut({
    shouldHandle: (event) => historyShortcut(event) !== null,
    onDown: (event) => {
      event.preventDefault();
      if (historyShortcut(event) === "undo") handleUndo();
      else handleRedo();
    },
    options: { preventInInputs: true },
  });

  const handleStepClick = (index: number) => {
    if (index === 0) {
      setCurrentStep(0);
      setAcceptError(null);
      return;
    }
    if (index === 1 && canContinueToPace) {
      setAcceptError(null);
      setCurrentStep(1);
    }
  };

  const showTracksProblem = (problem: TracksProblem) => {
    setActiveTrack(problem.trackIndex);
    setAcceptError(problem.message);
  };

  const handleContinue = () => {
    if (tracksProblem) {
      showTracksProblem(tracksProblem);
      return;
    }
    setAcceptError(null);
    setCurrentStep(1);
  };

  const handleAccept = () => {
    if (tracksProblem) {
      showTracksProblem(tracksProblem);
      setCurrentStep(0);
      return;
    }
    onAccept(
      drawnTracks.map(({ id, points }) => ({ points, ...pacing.paceOf(id) })),
      pacing.durationSec,
    );
  };

  return (
    <Modal
      className={styles.modal}
      backdropClassName={styles.backdrop}
      variant="primary"
      onClose={onClose}
      disablePointerClickOutside
    >
      <Flex direction="column" gap="lg" width="100%" data-testid="motion-path-edit-modal">
        <Stepper
          items={stepItems}
          selectedIndex={currentStep}
          onStepClick={handleStepClick}
        />

        {currentStep === 0 ? (
          <Flex direction="column" gap="lg" width="100%">
            <div className={styles.body}>
              <div className={styles.editor}>
                <div className={styles.tracksOverlay} data-testid="motion-path-tracks">
                  <div className={styles.tracksRow}>
                    {draftTracks.map((_, index) => (
                      <Button
                        key={index}
                        appearance={index === activeTrack ? "brand" : "neutral"}
                        hierarchy="secondary"
                        size="sm"
                        label={`Track ${index + 1}`}
                        onClick={() => setActiveTrack(index)}
                      />
                    ))}
                    {draftTracks.length < MOTION_PATH_MAX_TRACKS ? (
                      <Button
                        appearance="neutral"
                        hierarchy="plain"
                        size="sm"
                        label="Add track"
                        onClick={handleAddTrack}
                      />
                    ) : null}
                    {draftTracks.length > 1 ? (
                      <Button
                        appearance="neutral"
                        hierarchy="plain"
                        size="sm"
                        label="Remove track"
                        onClick={handleRemoveTrack}
                      />
                    ) : null}
                  </div>
                  {drawnTracks.length > 1 ? (
                    <p className={styles.tracksWarning} role="status">
                      {TRACKS_WARNING}
                    </p>
                  ) : null}
                </div>
                <MotionPathCanvas
                  imageUrl={imageUrl}
                  imageAspect={imageAspect}
                  points={activePoints}
                  colorIndex={colorIndexOf(draftTracks, activeTrack)}
                  otherTracks={otherTracks}
                  tool={activeTool}
                  onPointsChange={handlePointsChange}
                />
                <div className={styles.toolbarOverlay}>
                  <MotionPathToolbar
                    activeTool={activeTool}
                    onSelectTool={setSelectedTool}
                    onUndo={handleUndo}
                    onRedo={handleRedo}
                    onClear={handleClear}
                    canUndo={pastTracks.length > 0}
                    canRedo={futureTracks.length > 0}
                    canClear={activePoints.length > 0}
                  />
                </div>
              </div>

              {acceptError ? (
                <p className={styles.error} role="alert">
                  {acceptError}
                </p>
              ) : null}
            </div>

            <Flex justify="end" gap="md">
              <Button
                appearance="neutral"
                hierarchy="plain"
                size="md"
                label="Cancel"
                onClick={onClose}
              />
              <Button
                appearance="brand"
                hierarchy="primary"
                size="md"
                label="Continue"
                onClick={handleContinue}
              />
            </Flex>
          </Flex>
        ) : (
          <MotionPathPaceStep
            imageUrl={imageUrl}
            imageAspect={imageAspect}
            tracks={drawnTracks}
            pacing={pacing}
            durationOptions={durationOptions}
            onAccept={handleAccept}
            onCancel={onClose}
          />
        )}
      </Flex>
    </Modal>
  );
}
