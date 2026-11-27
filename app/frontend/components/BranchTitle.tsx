import { forkProductName } from '../../shared/fork-identity'
import styles from './BranchTitle.module.scss'

export function BranchTitle({ className = '' }: { className?: string }) {
  const platform = window.electronAPI?.platform
  return (
    <h1 className={`${styles.title} ${className}`}>
      {forkProductName(platform)}
      <span className={styles.subtitle}>
        {platform === 'darwin' ? 'Community branch with Apple silicon GPU support' : 'Community branch with AMD GPU support'}
      </span>
    </h1>
  )
}
