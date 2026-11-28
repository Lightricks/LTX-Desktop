import {
  type Announcements,
  DndContext,
  type DragEndEvent,
  KeyboardSensor,
  MouseSensor,
  TouchSensor,
  closestCenter,
  useSensor,
  useSensors,
} from "@dnd-kit/core";
import {
  SortableContext,
  rectSwappingStrategy,
  sortableKeyboardCoordinates,
} from "@dnd-kit/sortable";
import type { ReactNode } from "react";

/**
 * The drag and drop of the keyframe list. The rows must be in time order. A still dropped on
 * another one takes its place, and that one takes the place of the first. `onSwap` gets both ids.
 * The same sensors as the generation queue, so it works with a mouse, a finger and the keyboard.
 */
export function SortableKeyframeList({
  orderedIds,
  onSwap,
  children,
}: {
  /** The ids of the stills in time order. The place in this list is the number of the still. */
  orderedIds: readonly string[];
  onSwap: (activeId: string, overId: string) => void;
  children: ReactNode;
}) {
  const sensors = useSensors(
    useSensor(MouseSensor, { activationConstraint: { distance: 8 } }),
    useSensor(TouchSensor, { activationConstraint: { delay: 150, tolerance: 5 } }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates }),
  );
  const name = (id: string | number) => `keyframe ${orderedIds.indexOf(String(id)) + 1}`;
  // The default texts name the id of the still. A person needs its number.
  const announcements: Announcements = {
    onDragStart: ({ active }) => `Picked up ${name(active.id)}.`,
    onDragOver: ({ active, over }) =>
      over ? `${name(active.id)} is over ${name(over.id)}.` : undefined,
    onDragEnd: ({ active, over }) =>
      over && over.id !== active.id
        ? `${name(active.id)} swapped with ${name(over.id)}.`
        : `${name(active.id)} dropped where it was.`,
    onDragCancel: ({ active }) => `Move of ${name(active.id)} cancelled.`,
  };

  const handleDragEnd = ({ active, over }: DragEndEvent) => {
    if (over && active.id !== over.id) onSwap(String(active.id), String(over.id));
  };

  return (
    <DndContext
      sensors={sensors}
      collisionDetection={closestCenter}
      accessibility={{ announcements }}
      onDragEnd={handleDragEnd}
    >
      <SortableContext items={[...orderedIds]} strategy={rectSwappingStrategy}>
        {children}
      </SortableContext>
    </DndContext>
  );
}
