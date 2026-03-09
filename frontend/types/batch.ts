import type { QueueJob } from '../hooks/use-generation'

export interface BatchJobItem {
  type: 'video' | 'image'
  model: string
  params: Record<string, unknown>
}

export interface SweepAxis {
  param: string
  values: unknown[]
  mode: 'replace' | 'search_replace'
  search?: string
}

export interface SweepDefinition {
  base_type: 'video' | 'image'
  base_model: string
  base_params: Record<string, unknown>
  axes: SweepAxis[]
}

export interface PipelineStep {
  type: 'video' | 'image'
  model: string
  params: Record<string, unknown>
  auto_prompt: boolean
}

export interface PipelineDefinition {
  steps: PipelineStep[]
}

export interface BatchSubmitRequest {
  mode: 'list' | 'sweep' | 'pipeline'
  target: 'local' | 'cloud'
  jobs?: BatchJobItem[]
  sweep?: SweepDefinition
  pipeline?: PipelineDefinition
}

export interface BatchSubmitResponse {
  batch_id: string
  job_ids: string[]
  total_jobs: number
}

export interface BatchReport {
  batch_id: string
  total: number
  succeeded: number
  failed: number
  cancelled: number
  duration_seconds: number
  avg_job_seconds: number
  result_paths: string[]
  failed_indices: number[]
  sweep_axes: string[] | null
}

export interface BatchStatusResponse {
  batch_id: string
  total: number
  completed: number
  failed: number
  running: number
  queued: number
  cancelled: number
  jobs: QueueJob[]
  report: BatchReport | null
}
