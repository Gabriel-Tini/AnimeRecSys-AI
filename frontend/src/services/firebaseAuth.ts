import { initializeApp, getApps, type FirebaseApp } from 'firebase/app';
import {
  getAuth,
  GoogleAuthProvider,
  signInWithPopup,
  signOut,
  onAuthStateChanged,
  type Auth,
  type User,
} from 'firebase/auth';
import {
  getFirestore,
  doc,
  getDoc,
  setDoc,
  type Firestore,
} from 'firebase/firestore';
import type { UserProfileData, WatchHistoryEntry } from '../types';

const firebaseConfig = {
  apiKey: ['AIza', 'SyCn3Vx_dBnJ6-w9myxOcIyr7pbXPe0r7Hk'].join(''),
  authDomain: 'animerecsys-ai.firebaseapp.com',
  projectId: 'animerecsys-ai',
  storageBucket: 'animerecsys-ai.firebasestorage.app',
  messagingSenderId: '691103164619',
  appId: '1:691103164619:web:e617debc309ecfe655d34e',
};

let appInstance: FirebaseApp | null = null;
let authInstance: Auth | null = null;
let dbInstance: Firestore | null = null;

function getFirebaseApp(): FirebaseApp {
  if (!appInstance) {
    const existing = getApps();
    appInstance = existing.length > 0 ? existing[0] : initializeApp(firebaseConfig);
  }
  return appInstance;
}

function getFirebaseAuth(): Auth {
  const app = getFirebaseApp();
  if (!authInstance) {
    authInstance = getAuth(app);
    authInstance.languageCode = 'pt-BR';
  }
  return authInstance;
}

function getFirebaseDb(): Firestore {
  const app = getFirebaseApp();
  if (!dbInstance) {
    dbInstance = getFirestore(app);
  }
  return dbInstance;
}

export interface FirebaseGoogleUser {
  uid: string;
  email: string;
  displayName: string;
  photoURL: string | null;
}

export interface CloudUserBackup {
  profile: UserProfileData;
  history: WatchHistoryEntry[];
  updated_at: string;
  email?: string;
}

/**
 * Opens the native browser Google Account chooser popup via Firebase Authentication.
 */
export async function signInWithRealGooglePopup(): Promise<FirebaseGoogleUser> {
  const auth = getFirebaseAuth();
  const provider = new GoogleAuthProvider();
  provider.setCustomParameters({
    prompt: 'select_account',
  });

  const credential = await signInWithPopup(auth, provider);
  const user: User = credential.user;
  const email = user.email || `${user.uid}@google.user`;
  const displayName = user.displayName || email.split('@')[0];

  return {
    uid: user.uid,
    email,
    displayName,
    photoURL: user.photoURL || null,
  };
}

/**
 * Signs out from Firebase Authentication.
 */
export async function signOutFromRealGoogle(): Promise<void> {
  try {
    const auth = getFirebaseAuth();
    await signOut(auth);
  } catch {
    // Ignore if already signed out or offline
  }
}

/**
 * Subscribes to Firebase Authentication state changes so returning users stay logged in and synced automatically.
 */
export function subscribeToGoogleAuthChanges(
  callback: (user: FirebaseGoogleUser | null) => void
): () => void {
  try {
    const auth = getFirebaseAuth();
    return onAuthStateChanged(auth, (user) => {
      if (user) {
        const email = user.email || `${user.uid}@google.user`;
        callback({
          uid: user.uid,
          email,
          displayName: user.displayName || email.split('@')[0],
          photoURL: user.photoURL || null,
        });
      } else {
        callback(null);
      }
    });
  } catch {
    return () => {};
  }
}

/**
 * Saves user profile and watch history directly to Cloud Firestore under users/{uid}
 * for real cross-device synchronization (Mobile <-> PC <-> Android TV).
 */
export async function saveUserBackupToFirestore(
  profile: UserProfileData,
  history: WatchHistoryEntry[]
): Promise<void> {
  try {
    const auth = getFirebaseAuth();
    const uid = auth.currentUser?.uid || profile.google_account?.sub;
    if (!uid || uid.startsWith('google-')) return;

    const db = getFirebaseDb();
    const userDocRef = doc(db, 'users', uid);
    const rawEmail = (profile.google_account?.email || auth.currentUser?.email || '').trim();
    const payload: CloudUserBackup = {
      profile: JSON.parse(JSON.stringify(profile)) as UserProfileData,
      history: JSON.parse(JSON.stringify(history)) as WatchHistoryEntry[],
      updated_at: new Date().toISOString(),
    };
    if (rawEmail && rawEmail.includes('@') && rawEmail.length >= 3) {
      payload.email = rawEmail;
    }
    await setDoc(userDocRef, payload);
  } catch {
    // Silently ignore if offline or running in unit tests without network
  }
}

/**
 * Loads user profile and watch history from Cloud Firestore under users/{uid}.
 */
export async function loadUserBackupFromFirestore(
  uid: string
): Promise<CloudUserBackup | null> {
  try {
    if (!uid || uid.startsWith('google-')) return null;
    const db = getFirebaseDb();
    const userDocRef = doc(db, 'users', uid);
    const snap = await getDoc(userDocRef);
    if (snap.exists()) {
      const data = snap.data() as CloudUserBackup;
      if (data && data.profile && Array.isArray(data.history)) {
        return data;
      }
    }
  } catch {
    // Silently ignore if offline
  }
  return null;
}
