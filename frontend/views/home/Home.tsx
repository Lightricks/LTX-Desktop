import { useNavigate } from 'react-router'
import { Text } from '@/ds/Text/Text'
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
        <Text
          as="h1"
          variant="display"
          size="sm"
          align="center"
          className={styles.headline}
        >
          Explore endless creative solutions
        </Text>
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
