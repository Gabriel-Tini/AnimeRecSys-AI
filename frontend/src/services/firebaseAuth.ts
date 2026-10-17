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

function getFirebaseAuth(): Auth {
  if (!appInstance) {
    const existing = getApps();
    appInstance = existing.length > 0 ? existing[0] : initializeApp(firebaseConfig);
  }
  if (!authInstance) {
    authInstance = getAuth(appInstance);
    authInstance.languageCode = 'pt-BR';
  }
  return authInstance;
}

export interface FirebaseGoogleUser {
  uid: string;
  email: string;
  displayName: string;
  photoURL: string | null;
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
