import { useNavigate } from 'react-router'
import { BranchTitle } from '../../components/BranchTitle'
// eslint-disable-next-line no-restricted-imports
import { QuickSearchLaunchField } from '@/ltx-io/components/QuickSearch/QuickSearchLaunchField'
// eslint-disable-next-line no-restricted-imports
import { useQuickSearch } from '@/ltx-io/components/QuickSearch/QuickSearchContext'
import { HomeGallery } from '../../components/home/HomeGallery'
import styles from './Home.module.scss'

export function Home() {
  const navigate = useNavigate()
  const { isOpen: isQuickSearchOpen, open: openQuickSearch } = useQuickSearch()

  return (
    <div className={styles.page}>
      <div className={styles.hero}>
        <BranchTitle className={styles.headline} />
        <div className={styles.searchBarSlot}>
          <QuickSearchLaunchField
            hidden={isQuickSearchOpen}
            onActivate={openQuickSearch}
          />
        </div>
      </div>
      <HomeGallery onOpen={(path) => navigate(path)} />
    </div>
  )
}
