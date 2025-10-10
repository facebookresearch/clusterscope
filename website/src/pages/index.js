import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import ThemedImage from '@theme/ThemedImage';

import Heading from '@theme/Heading';
import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <ThemedImage
          alt="Clusterscope Logo"
          sources={{
            light: 'img/clusterscope_long_white.svg',
            dark: 'img/clusterscope_long_dark_blue.svg',
          }}
          style={{maxWidth: '50%', minWidth: '250px'}}
        />
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div
          className={styles.buttons}
        >
          <Link
            style={{
              backgroundColor: 'var(--btn-green-bg)',
              borderColor: 'var(--btn-green-bg)',
              color: 'var(--btn-green-text)'
            }}
            className="button button--secondary button--lg"
            to="/docs/getting_started">
            Cluterscope Getting Started
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`${siteConfig.title}`}
      description="Clusterscope is a CLI and python library to extract information from HPC Clusters and Jobs.">
      <HomepageHeader />
      <main>
        <HomepageFeatures />
      </main>
    </Layout>
  );
}
