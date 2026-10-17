/**
 * frontend/src/mocks/fixtures.ts
 * Canonical fallback data mirrored from .antigravity/FIXTURES.json.
 * IMPORTANT: Never import this file directly inside UI components.
 * All access must go exclusively through frontend/src/services/api.ts.
 */

import type {
  AnimeTag,
  RecommendationPayload,
  SemanticTuningResponse,
  UserProfileData,
  WatchHistoryEntry,
} from '../types';

export interface RawAnimeFixture {
  id: number;
  title_romaji: string;
  title_english?: string;
  title_native?: string;
  cover_image: string;
  banner_image?: string;
  format: string;
  episodes?: number;
  duration?: number;
  status: string;
  season_year?: number;
  average_score?: number;
  popularity: number;
  genres: string[];
  tags: AnimeTag[];
  description?: string;
  has_spoiler_tags?: boolean;
}

export const TAGS_VOCABULARY: AnimeTag[] = [
  { id: 1, name: 'Action', rank: 95, is_media_spoiler: false },
  { id: 2, name: 'Adventure', rank: 90, is_media_spoiler: false },
  { id: 3, name: 'Fantasy', rank: 92, is_media_spoiler: false },
  { id: 4, name: 'Sci-Fi', rank: 88, is_media_spoiler: false },
  { id: 5, name: 'Cyberpunk', rank: 86, is_media_spoiler: false },
  { id: 6, name: 'Time Manipulation', rank: 89, is_media_spoiler: false },
  { id: 7, name: 'Psychological', rank: 91, is_media_spoiler: false },
  { id: 8, name: 'Mystery', rank: 87, is_media_spoiler: false },
  { id: 9, name: 'Supernatural', rank: 85, is_media_spoiler: false },
  { id: 10, name: 'Shounen', rank: 94, is_media_spoiler: false },
  { id: 11, name: 'Plot Twist', rank: 93, is_media_spoiler: true },
  { id: 12, name: 'Betrayal', rank: 84, is_media_spoiler: true },
  { id: 13, name: 'Tragedy', rank: 89, is_media_spoiler: true },
  { id: 14, name: 'Anti-Hero', rank: 87, is_media_spoiler: false },
  { id: 15, name: 'Conspiracy', rank: 82, is_media_spoiler: true },
  { id: 16, name: 'Memory Manipulation', rank: 80, is_media_spoiler: true },
  { id: 17, name: 'Philosophy', rank: 83, is_media_spoiler: false },
  { id: 18, name: 'Coming of Age', rank: 81, is_media_spoiler: false },
  { id: 19, name: 'Slice of Life', rank: 78, is_media_spoiler: false },
  { id: 20, name: 'Music', rank: 85, is_media_spoiler: false },
];

export const SAMPLE_ANIMES_FULL: RawAnimeFixture[] = [
  {
    id: 9253,
    title_romaji: 'Steins;Gate',
    title_english: 'Steins;Gate',
    title_native: 'シュタインズ・ゲート',
    cover_image: 'https://s4.anilist.co/file/anilistcdn/media/anime/cover/large/bx9253-tIUXF2gfU8Sg.jpg',
    banner_image: 'https://s4.anilist.co/file/anilistcdn/media/anime/banner/n9253-JIhmKgBKsWUN.jpg',
    format: 'TV',
    episodes: 24,
    duration: 24,
    status: 'FINISHED',
    season_year: 2011,
    average_score: 90,
    popularity: 320400,
    genres: ['Drama', 'Sci-Fi', 'Suspense'],
    tags: [
      { id: 6, name: 'Time Manipulation', rank: 98, is_media_spoiler: false },
      { id: 7, name: 'Psychological', rank: 92, is_media_spoiler: false },
      { id: 11, name: 'Plot Twist', rank: 96, is_media_spoiler: true },
      { id: 13, name: 'Tragedy', rank: 88, is_media_spoiler: true },
      { id: 15, name: 'Conspiracy', rank: 85, is_media_spoiler: true },
    ],
  },
  {
    id: 16498,
    title_romaji: 'Shingeki no Kyojin',
    title_english: 'Attack on Titan',
    title_native: '進撃の巨人',
    cover_image: 'https://s4.anilist.co/file/anilistcdn/media/anime/cover/large/bx16498-buvcRTBx4NSm.jpg',
    banner_image: 'https://s4.anilist.co/file/anilistcdn/media/anime/banner/16498-8jpFCOcDmneX.jpg',
    format: 'TV',
    episodes: 25,
    duration: 24,
    status: 'FINISHED',
    season_year: 2013,
    average_score: 85,
    popularity: 540200,
    genres: ['Action', 'Drama', 'Fantasy', 'Mystery'],
    tags: [
      { id: 1, name: 'Action', rank: 95, is_media_spoiler: false },
      { id: 11, name: 'Plot Twist', rank: 94, is_media_spoiler: true },
      { id: 12, name: 'Betrayal', rank: 91, is_media_spoiler: true },
      { id: 13, name: 'Tragedy', rank: 93, is_media_spoiler: true },
      { id: 14, name: 'Anti-Hero', rank: 89, is_media_spoiler: false },
    ],
  },
  {
    id: 154587,
    title_romaji: 'Sousou no Frieren',
    title_english: "Frieren: Beyond Journey's End",
    title_native: '葬送のフリーレン',
    cover_image: 'https://s4.anilist.co/file/anilistcdn/media/anime/cover/large/bx154587-qQTzQnEJJ3oB.jpg',
    banner_image: 'https://s4.anilist.co/file/anilistcdn/media/anime/banner/154587-ivXNJ23SM1xB.jpg',
    format: 'TV',
    episodes: 28,
    duration: 24,
    status: 'FINISHED',
    season_year: 2023,
    average_score: 93,
    popularity: 280500,
    genres: ['Adventure', 'Drama', 'Fantasy'],
    tags: [
      { id: 2, name: 'Adventure', rank: 96, is_media_spoiler: false },
      { id: 3, name: 'Fantasy', rank: 95, is_media_spoiler: false },
      { id: 17, name: 'Philosophy', rank: 88, is_media_spoiler: false },
      { id: 18, name: 'Coming of Age', rank: 84, is_media_spoiler: false },
    ],
  },
  {
    id: 151807,
    title_romaji: 'Ore dake Level Up na Ken',
    title_english: 'Solo Leveling',
    title_native: '俺だけレベルアップな件',
    cover_image: 'https://s4.anilist.co/file/anilistcdn/media/anime/cover/large/bx151807-it355ZgzquUd.png',
    banner_image: 'https://s4.anilist.co/file/anilistcdn/media/anime/banner/151807-37yfQA3ym8PA.jpg',
    format: 'TV',
    episodes: 12,
    duration: 24,
    status: 'FINISHED',
    season_year: 2024,
    average_score: 83,
    popularity: 310000,
    genres: ['Action', 'Adventure', 'Fantasy'],
    tags: [
      { id: 1, name: 'Action', rank: 96, is_media_spoiler: false },
      { id: 9, name: 'Supernatural', rank: 90, is_media_spoiler: false },
      { id: 14, name: 'Anti-Hero', rank: 85, is_media_spoiler: false },
    ],
  },
  {
    id: 126546,
    title_romaji: 'Cyberpunk: Edgerunners',
    title_english: 'Cyberpunk: Edgerunners',
    title_native: 'サイバーパンク エッジランナーズ',
    cover_image: 'https://s4.anilist.co/file/anilistcdn/media/anime/cover/large/bx126546-Jujx9OvKxLzA.jpg',
    banner_image: 'https://s4.anilist.co/file/anilistcdn/media/anime/banner/126546-eSjc83C90h24.jpg',
    format: 'ONA',
    episodes: 10,
    duration: 24,
    status: 'FINISHED',
    season_year: 2022,
    average_score: 86,
    popularity: 290000,
    genres: ['Action', 'Sci-Fi'],
    tags: [
      { id: 1, name: 'Action', rank: 95, is_media_spoiler: false },
      { id: 4, name: 'Sci-Fi', rank: 92, is_media_spoiler: false },
      { id: 5, name: 'Cyberpunk', rank: 98, is_media_spoiler: false },
      { id: 13, name: 'Tragedy', rank: 94, is_media_spoiler: true },
    ],
  },
  {
    id: 145064,
    title_romaji: 'Bocchi the Rock!',
    title_english: 'Bocchi the Rock!',
    title_native: 'ぼっち・ざ・ろっく！',
    cover_image: 'https://s4.anilist.co/file/anilistcdn/media/anime/cover/large/bx145064-hSNRJM03pvv1.jpg',
    banner_image: 'https://s4.anilist.co/file/anilistcdn/media/anime/banner/145064-esDtAY2He7sk.jpg',
    format: 'TV',
    episodes: 12,
    duration: 24,
    status: 'FINISHED',
    season_year: 2022,
    average_score: 88,
    popularity: 240000,
    genres: ['Comedy', 'Music', 'Slice of Life'],
    tags: [
      { id: 18, name: 'Coming of Age', rank: 85, is_media_spoiler: false },
      { id: 19, name: 'Slice of Life', rank: 92, is_media_spoiler: false },
      { id: 20, name: 'Music', rank: 97, is_media_spoiler: false },
    ],
  },
];

export const MOCK_USER_PROFILE: UserProfileData = {
  user_id: 'default_user',
  updated_at: '2026-10-14T20:00:00Z',
  total_ratings: 2,
  feature_weights: {
    'Sci-Fi': 0.428,
    'Time Manipulation': 0.512,
    Psychological: 0.445,
    'Plot Twist': 0.39,
    Tragedy: 0.315,
    Cyberpunk: 0.28,
    Action: 0.12,
    'Slice of Life': -0.05,
  },
  hyperparameters: {
    eta: 0.15,
    alpha: 0.25,
    gamma: 0.98,
    lambda_mmr: 0.5,
  },
};

export const MOCK_WATCH_HISTORY: WatchHistoryEntry[] = [
  {
    id: 'wh-001',
    anime_id: 9253,
    anime_title: 'Steins;Gate',
    rating: 10.0,
    origin_track: 'onboarding',
    user_review_raw: 'Obra-prima absoluta, o plot twist no episódio 22 e as viagens no tempo foram incríveis.',
    qualitative_tuning: {
      identified_tags: [
        {
          tag_name: 'Time Manipulation',
          impact_multiplier: 0.95,
          confidence: 0.98,
          rationale: 'Elogio enfático às mecânicas de viagem no tempo.',
        },
        {
          tag_name: 'Plot Twist',
          impact_multiplier: 0.9,
          confidence: 0.95,
          rationale: 'Citação direta do plot twist de clímax.',
        },
        {
          tag_name: 'Psychological',
          impact_multiplier: 0.7,
          confidence: 0.85,
          rationale: 'Elemento intrínseco aos episódios citados.',
        },
      ],
      public_safe_reason:
        'Você demonstrou forte afinidade por ficção científica com manipulação temporal e suspense psicológico denso.',
    },
    timestamp: '2026-10-14T19:30:00Z',
  },
  {
    id: 'wh-002',
    anime_id: 16498,
    anime_title: 'Attack on Titan',
    rating: 9.0,
    origin_track: 'affinity',
    user_review_raw: 'Excelente construção de mistério e desespero, animação excelente.',
    qualitative_tuning: {
      identified_tags: [
        {
          tag_name: 'Mystery',
          impact_multiplier: 0.8,
          confidence: 0.9,
          rationale: 'Apreço pela construção de mistério no enredo.',
        },
        {
          tag_name: 'Action',
          impact_multiplier: 0.6,
          confidence: 0.85,
          rationale: 'Elogio à qualidade e ritmo de combate.',
        },
      ],
      public_safe_reason: 'Você valorizou mistérios de alta intensidade e combates com grande peso dramático.',
    },
    timestamp: '2026-10-14T19:45:00Z',
  },
];

export const MOCK_GEMINI_RESPONSE: SemanticTuningResponse = {
  identified_tags: [
    {
      tag_name: 'Time Manipulation',
      impact_multiplier: 0.95,
      confidence: 0.98,
      rationale: 'O usuário elogiou fortemente a mecânica de loops temporais.',
    },
    {
      tag_name: 'Plot Twist',
      impact_multiplier: 0.85,
      confidence: 0.92,
      rationale: 'Reconhecimento explícito de reviravoltas na trama.',
    },
    {
      tag_name: 'Psychological',
      impact_multiplier: 0.75,
      confidence: 0.88,
      rationale: 'Conexão com a tensão mental e suspense enfrentado pelos personagens.',
    },
  ],
  public_safe_reason:
    'Você demonstrou grande entusiasmo por ficção científica instigante com suspense psicológico e narrativa não-linear.',
};

export const MOCK_RECOMMENDATION_RESPONSE: RecommendationPayload = {
  affinity_track: [
    {
      id: 126546,
      title_romaji: 'Cyberpunk: Edgerunners',
      title_english: 'Cyberpunk: Edgerunners',
      cover_image: 'https://s4.anilist.co/file/anilistcdn/media/anime/cover/large/bx126546-Jujx9OvKxLzA.jpg',
      banner_image: 'https://s4.anilist.co/file/anilistcdn/media/anime/banner/126546-eSjc83C90h24.jpg',
      format: 'ONA',
      episodes: 10,
      status: 'FINISHED',
      season_year: 2022,
      average_score: 86,
      genres: ['Action', 'Sci-Fi'],
      tags: ['Action', 'Sci-Fi', 'Cyberpunk'],
      compatibility_score: 0.942,
      origin_track: 'affinity',
    },
    {
      id: 151807,
      title_romaji: 'Ore dake Level Up na Ken',
      title_english: 'Solo Leveling',
      cover_image: 'https://s4.anilist.co/file/anilistcdn/media/anime/cover/large/bx151807-it355ZgzquUd.png',
      banner_image: 'https://s4.anilist.co/file/anilistcdn/media/anime/banner/151807-37yfQA3ym8PA.jpg',
      format: 'TV',
      episodes: 12,
      status: 'FINISHED',
      season_year: 2024,
      average_score: 83,
      genres: ['Action', 'Adventure', 'Fantasy'],
      tags: ['Action', 'Supernatural', 'Anti-Hero'],
      compatibility_score: 0.875,
      origin_track: 'affinity',
    },
  ],
  discovery_track: [
    {
      id: 154587,
      title_romaji: 'Sousou no Frieren',
      title_english: "Frieren: Beyond Journey's End",
      cover_image: 'https://s4.anilist.co/file/anilistcdn/media/anime/cover/large/bx154587-qQTzQnEJJ3oB.jpg',
      banner_image: 'https://s4.anilist.co/file/anilistcdn/media/anime/banner/154587-ivXNJ23SM1xB.jpg',
      format: 'TV',
      episodes: 28,
      status: 'FINISHED',
      season_year: 2023,
      average_score: 93,
      genres: ['Adventure', 'Drama', 'Fantasy'],
      tags: ['Adventure', 'Fantasy', 'Philosophy', 'Coming of Age'],
      compatibility_score: 0.765,
      origin_track: 'discovery',
    },
    {
      id: 145064,
      title_romaji: 'Bocchi the Rock!',
      title_english: 'Bocchi the Rock!',
      cover_image: 'https://s4.anilist.co/file/anilistcdn/media/anime/cover/large/bx145064-hSNRJM03pvv1.jpg',
      banner_image: 'https://s4.anilist.co/file/anilistcdn/media/anime/banner/145064-esDtAY2He7sk.jpg',
      format: 'TV',
      episodes: 12,
      status: 'FINISHED',
      season_year: 2022,
      average_score: 88,
      genres: ['Comedy', 'Music', 'Slice of Life'],
      tags: ['Coming of Age', 'Slice of Life', 'Music'],
      compatibility_score: 0.69,
      origin_track: 'discovery',
    },
  ],
  generated_at: '2026-10-14T20:10:00Z',
};
