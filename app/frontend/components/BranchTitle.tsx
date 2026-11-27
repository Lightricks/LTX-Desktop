import { forkProductName } from '../../shared/fork-identity'
import styles from './BranchTitle.module.scss'

const forkSubtitle: Record<string, string> = {
  darwin: 'Community fork with Apple silicon GPU support',
  linux: 'Community fork with NVIDIA CUDA support',
  win32: 'Community fork with AMD and NVIDIA GPU support',
}

export function BranchTitle({ className = '' }: { className?: string }) {
  const platform = window.electronAPI?.platform
  return (
    <h1 className={`${styles.title} ${className}`}>
      {forkProductName(platform)}
      <span className={styles.subtitle}>
        {forkSubtitle[platform ?? ''] ?? 'Community fork of LTX Desktop'}
      </span>
    </h1>
  )
}
