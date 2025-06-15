import './globals.css';
import type { Metadata } from 'next';
import { Geist, Geist_Mono } from 'next/font/google';
import { I18nProvider } from '../src/components/I18nProvider';
import { appWithTranslation } from 'next-i18next';

const geistSans = Geist({ variable: '--font-geist-sans', subsets: ['latin'] });
const geistMono = Geist_Mono({ variable: '--font-geist-mono', subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'LeafLens',
  description: 'Offline plant disease detector',
};

function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased`}>
        <I18nProvider>
          {children}
        </I18nProvider>
        {children}
      </body>
    </html>
  );
}

export default RootLayout;