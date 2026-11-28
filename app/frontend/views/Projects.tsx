import { useEffect, useMemo, useRef, useState } from "react";
import { Plus, Folder, MoreVertical, Trash2, Pencil } from "lucide-react";
import { useNavigate } from "react-router";
import { Button } from "@ds/Button/Button";
import { Text } from "@ds/Text/Text";
import { TextField } from "@ds/TextField/TextField";
import { useProjects } from "../contexts/ProjectContext";
import { openProject } from "../lib/project-navigation";
import { pathToFileUrl } from "../lib/file-url";
import type { Project } from "../types/project-model";
import { useProjectReferencesMigration } from "../hooks/useProjectReferencesMigration";

function formatDate(timestamp: number): string {
  const date = new Date(timestamp);
  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function ProjectCard({
  project,
  onOpen,
  onDelete,
  onRename,
}: {
  project: Project;
  onOpen: () => void;
  onDelete: () => void;
  onRename: () => void;
}) {
  const [showMenu, setShowMenu] = useState(false);
  const [imgError, setImgError] = useState(false);

  // Keep existing representative selection logic: prefer first image, else first asset.
  const representativeAsset =
    project.assets.find((a) => a.type === "image") || project.assets[0] || null;
  const representativeUrl = representativeAsset?.path
    ? pathToFileUrl(representativeAsset.path)
    : null;
  const representativeBigThumbnailUrl = representativeAsset?.bigThumbnailPath
    ? pathToFileUrl(representativeAsset.bigThumbnailPath)
    : null;

  return (
    <div
      className="group relative bg-surface-primary rounded-[var(--radius-lg)] overflow-hidden border border-separator-secondary hover:border-separator transition-colors cursor-pointer"
      onClick={onOpen}
    >
      <div className="aspect-video bg-surface-secondary flex items-center justify-center relative overflow-hidden">
        {representativeAsset && !imgError ? (
          representativeAsset.type === "video" ? (
            representativeBigThumbnailUrl ? (
              <img
                src={representativeBigThumbnailUrl}
                alt={project.name}
                className="w-full h-full object-cover"
                onError={() => setImgError(true)}
              />
            ) : representativeUrl ? (
              <video
                src={representativeUrl}
                className="w-full h-full object-cover"
                muted
                preload="metadata"
                onError={() => setImgError(true)}
              />
            ) : (
              <Folder className="h-12 w-12 text-fg-tertiary" />
            )
          ) : representativeUrl ? (
            <img
              src={representativeUrl}
              alt={project.name}
              className="w-full h-full object-cover"
              onError={() => setImgError(true)}
            />
          ) : (
            <Folder className="h-12 w-12 text-fg-tertiary" />
          )
        ) : (
          <Folder className="h-12 w-12 text-fg-tertiary" />
        )}
        {/* Hover overlay */}
        <div className="absolute inset-0 bg-black/30 opacity-0 group-hover:opacity-100 transition-opacity" />
      </div>

      <div className="p-3">
        <Text as="h3" variant="label" size="xl" shouldTruncate>
          {project.name}
        </Text>
        <Text as="p" variant="body" size="sm" className="mt-1 text-fg-secondary">
          {formatDate(project.updatedAt)}
        </Text>
      </div>

      <button
        onClick={(e) => {
          e.stopPropagation();
          setShowMenu(!showMenu);
        }}
        className="absolute top-2 right-2 p-1.5 rounded bg-surface-overlay opacity-0 group-hover:opacity-100 transition-opacity"
      >
        <MoreVertical className="h-4 w-4 text-fg-white" />
      </button>

      {showMenu && (
        <div
          className="absolute top-10 right-2 bg-surface-primary rounded-[var(--radius-lg)] shadow-lg border border-separator py-1 z-10 min-w-[120px]"
          onClick={(e) => e.stopPropagation()}
        >
          <button
            onClick={() => {
              onRename();
              setShowMenu(false);
            }}
            className="w-full px-3 py-2 text-left text-fg-primary hover:bg-action flex items-center gap-2"
          >
            <Pencil className="h-4 w-4" />
            <Text as="span" variant="body" size="lg">Rename</Text>
          </button>
          <button
            onClick={() => {
              onDelete();
              setShowMenu(false);
            }}
            className="w-full px-3 py-2 text-left text-fg-danger hover:bg-action flex items-center gap-2"
          >
            <Trash2 className="h-4 w-4" />
            <Text as="span" variant="body" size="lg">Delete</Text>
          </button>
        </div>
      )}
    </div>
  );
}

function NewProjectCard({ onCreate }: { onCreate: () => void }) {
  return (
    <button
      type="button"
      className="group relative bg-surface-primary rounded-[var(--radius-lg)] overflow-hidden border border-separator-secondary hover:border-separator transition-colors text-left"
      onClick={onCreate}
    >
      <div className="aspect-video bg-surface-secondary flex items-center justify-center">
        <Plus className="h-12 w-12 text-fg-tertiary" />
      </div>
      <div className="p-3">
        <Text as="h3" variant="label" size="xl" shouldTruncate>
          New Project
        </Text>
        <Text as="p" variant="body" size="sm" className="mt-1 text-fg-secondary">
          Create a timeline project
        </Text>
      </div>
    </button>
  );
}

export function ProjectsPage() {
  const {
    projectIds,
    getProject,
    createProject,
    deleteProject,
    renameProject,
    activateProject,
    setCurrentTab,
  } = useProjects();
  const navigate = useNavigate();

  const handleOpenProject = (projectId: string) => {
    openProject(projectId, { activateProject, setCurrentTab, navigate });
  };

  const { migrationStatus, migrateProjects } = useProjectReferencesMigration();
  const [isCreating, setIsCreating] = useState(false);
  const [newProjectName, setNewProjectName] = useState("");
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const migrationStartedRef = useRef(false);

  useEffect(() => {
    if (migrationStatus.status !== "needed" || migrationStartedRef.current)
      return;
    migrationStartedRef.current = true;
    void migrateProjects();
  }, [migrateProjects, migrationStatus.status]);

  const projects = useMemo(
    () =>
      projectIds
        .map((projectId) => getProject(projectId))
        .filter((project): project is Project => project !== null),
    [getProject, projectIds],
  );

  const handleCreateProject = () => {
    if (newProjectName.trim()) {
      const project = createProject(newProjectName.trim());
      setNewProjectName("");
      setIsCreating(false);
      handleOpenProject(project.id);
    }
  };

  const handleRenameProject = (id: string, currentName: string) => {
    setRenamingId(id);
    setRenameValue(currentName);
  };

  const submitRename = () => {
    if (renamingId && renameValue.trim()) {
      renameProject(renamingId, renameValue.trim());
    }
    setRenamingId(null);
    setRenameValue("");
  };

  if (
    migrationStatus.status === "needed" ||
    migrationStatus.status === "inProgress"
  ) {
    const progressPct =
      migrationStatus.status === "inProgress" ? migrationStatus.ratio * 100 : 0;

    return (
      <div className="h-full flex items-center justify-center">
        <div className="w-[360px]">
          <Text as="p" variant="body" size="lg" align="center" className="mb-4 text-fg-secondary">
            Migrating project references...
          </Text>
          <div className="h-2 w-full rounded-full bg-action overflow-hidden">
            <div
              className="h-full bg-brand transition-all duration-150"
              style={{ width: `${Math.max(0, Math.min(100, progressPct))}%` }}
            />
          </div>
        </div>
      </div>
    );
  }

  return (
    <>
      <div className="pb-8">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          <NewProjectCard onCreate={() => setIsCreating(true)} />
          {projects.map((project) => (
            <ProjectCard
              key={project.id}
              project={project}
              onOpen={() => handleOpenProject(project.id)}
              onDelete={() => {
                if (confirm(`Delete "${project.name}"?`)) {
                  deleteProject(project.id);
                }
              }}
              onRename={() => handleRenameProject(project.id, project.name)}
            />
          ))}
        </div>
      </div>

      {isCreating && (
        <div className="fixed inset-0 bg-surface-overlay flex items-center justify-center z-50">
          <form
            className="bg-surface-primary rounded-[var(--radius-lg)] p-6 w-full max-w-md border border-separator"
            onSubmit={(e) => {
              e.preventDefault();
              handleCreateProject();
            }}
          >
            <Text as="h2" variant="heading" size="lg" className="mb-4">
              Create New Project
            </Text>
            <TextField
              value={newProjectName}
              onInputChange={setNewProjectName}
              placeholder="Project name"
              autoFocus
            />
            <div className="flex justify-end gap-3 mt-6">
              <Button
                appearance="neutral"
                hierarchy="secondary"
                size="md"
                label="Cancel"
                onClick={() => {
                  setIsCreating(false);
                  setNewProjectName("");
                }}
              />
              <Button
                appearance="brand"
                hierarchy="primary"
                size="md"
                label="Create"
                type="submit"
                disabled={!newProjectName.trim()}
              />
            </div>
          </form>
        </div>
      )}

      {renamingId && (
        <div className="fixed inset-0 bg-surface-overlay flex items-center justify-center z-50">
          <form
            className="bg-surface-primary rounded-[var(--radius-lg)] p-6 w-full max-w-md border border-separator"
            onSubmit={(e) => {
              e.preventDefault();
              submitRename();
            }}
          >
            <Text as="h2" variant="heading" size="lg" className="mb-4">
              Rename Project
            </Text>
            <TextField
              value={renameValue}
              onInputChange={setRenameValue}
              placeholder="Project name"
              autoFocus
            />
            <div className="flex justify-end gap-3 mt-6">
              <Button
                appearance="neutral"
                hierarchy="secondary"
                size="md"
                label="Cancel"
                onClick={() => {
                  setRenamingId(null);
                  setRenameValue("");
                }}
              />
              <Button
                appearance="brand"
                hierarchy="primary"
                size="md"
                label="Save"
                type="submit"
                disabled={!renameValue.trim()}
              />
            </div>
          </form>
        </div>
      )}
    </>
  );
}
