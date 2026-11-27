import { productName } from '../../package.json'
import styles from './BranchTitle.module.scss'

export function BranchTitle({ className = '' }: { className?: string }) {
  return (
    <h1 className={`${styles.title} ${className}`}>
      {productName}
      <span className={styles.subtitle}>Community branch with AMD GPU support</span>
    </h1>
  )
}
