# AnimeRecSys AI 🚀
> Sistema Inteligente de Recomendação Híbrida de Animes com Aprendizado Vetorial em Tempo Real, Prevenção R3 de Spoilers e Interface Multiplataforma (Web, Mobile & Android TV).

![AnimeRecSys AI Platform](https://img.shields.io/badge/Platform-Web%20%7C%20Mobile%20%7C%20Android%20TV-indigo?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18.3-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![Vite](https://img.shields.io/badge/Vite-5.4-646CFF?style=for-the-badge&logo=vite&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3.4-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)
![Firebase](https://img.shields.io/badge/Firebase-Hosting-FFCA28?style=for-the-badge&logo=firebase&logoColor=black)

---

## 🌟 Visão Geral

O **AnimeRecSys AI** é uma plataforma completa e moderna de recomendação de animes orientada a dados. Diferente de sistemas tradicionais de recomendação baseados apenas em filtros estáticos ou médias populacionais, o AnimeRecSys modela cada usuário como um vetor dinâmico de alta dimensionalidade em um espaço semântico latente, adaptando pesos e preferências a cada interação imediata.

---

## 🧠 Núcleo Algorítmico & Funcionalidades

### 1. Recomendação Híbrida Dual-Track (80 / 20)
* **80% Trilha de Afinidade (Exploitation)**:
  * Projeção por Similaridade de Cosseno entre o vetor do usuário (normalizado via norma L2 $\|\vec{u}\|_2 = 1.0$) e o catálogo.
  * Ponderação temporal com decaimento exponencial gama ($\gamma = 0.95$), garantindo que preferências recentes tenham relevância proporcional ao seu timing.
* **20% Trilha de Descoberta (Exploration com MMR)**:
  * Aplicação do algoritmo **Maximal Marginal Relevance (MMR)** com fator de diversidade $\lambda = 0.65$.
  * Elimina a "bolha de recomendação", sugerindo obras aclamadas de gêneros correlatos sem desviar da identidade central do usuário.

### 2. Proteção R3 Contra Spoilers em Níveis Defensivos
* **Nível 1 (Backend Pipeline)**: Varredura de tags oficiais no AniList marcadas com flag booleana `isMediaSpoiler`.
* **Nível 2 (LLM / Motor Semântico)**: Algoritmo de filtragem contextual que remove revelações de enredo em sinopses antes de serem servidas à API.
* **Nível 3 (Frontend DOM Blur Guard)**: Mascaramento visual com gradiente blur dinâmico na sinopse. O usuário pode optar conscientemente por revelar ou manter protegido.

### 3. Catálogo Rico e Tradução em Português (pt-BR)
* **960+ animes catalogados** com metadados detalhados (título romaji, título ocidental em inglês, episódios, status, notas da comunidade, gêneros e tags semânticas).
* **Sinopses traduzidas** e limpas de marcações HTML brutas.
* **Cache Permanente de Mídia**: Sistema de redundância que armazena localmente posters e banners em `/api/media/cover/{id}`, blindando a plataforma contra expirações de CDN da AniList e servindo vetores SVG escuros como fallback à prova de falhas.

### 4. Proteções de Integridade de Dados
* **Bloqueio de Avaliações Duplicadas**: Status HTTP 409 e desativação no frontend para animes já presentes no histórico, evitando distorções no cálculo vetorial.
* **Área "Minha Lista" com Edição Ativa**: Permite editar notas e justificativas, recalculando em tempo real os pesos no perfil.
* **Sincronização Desacoplada**:
  * **Google Account**: Salva e sincroniza preferências, favoritos e bloqueios sem interferir na sessão local.
  * **AniList Sync**: Sincronização assíncrona de histórico de assistidos, lista de interesse e notas pessoais.

---

## 📱 Responsividade Multiplataforma

A interface foi projetada para se comportar com elegância e performance em qualquer tela:

* 💻 **Desktop**: Sidebar lateral ampla com acesso rápido, pesquisa Raycast (`Ctrl + K`), visualização em prateleiras e detalhes em modal imersivo.
* 📱 **Mobile (Smartphones)**:
  * Conversão da sidebar em **Bottom Navigation Bar nativa** com ícones táteis e badges de contagem.
  * Áreas seguras com padding adaptado para que nada fique oculto sob a barra de navegação.
  * Cabeçalho responsivo com ajuste automático dos botões de busca, novidades e login.
* 📺 **Android TV / Smart TVs (10-Foot UI)**:
  * Suporte nativo à navegação por **Controle Remoto (D-Pad)** com atributos `tabIndex={0}`.
  * Foco visual em alto contraste (`focus-visible:ring-4 focus-visible:ring-indigo-500 focus-visible:scale-105`).
  * Abertura direta via teclas `Enter`, `Espaço` ou botão central do controle.

---

## 🏗️ Arquitetura do Projeto

```
AnimeRecSys/
├── backend/                  # API Backend Assíncrona (FastAPI)
│   ├── app/
│   │   ├── api/routes.py     # Endpoints REST (/catalog, /recommendations, /rate, /media, etc.)
│   │   ├── core/recommender.py # Motor matemático L2, Cosseno e MMR
│   │   └── services/         # AniList GraphQL, Cache e Tradução
│   └── main.py               # Entrada principal da aplicação ASGI
├── frontend/                 # Frontend Moderno SPA (React 18 + Vite + TailwindCSS)
│   ├── src/
│   │   ├── components/       # AnimeCard, Navbar, DetailModal, SearchBar, etc.
│   │   ├── services/api.ts   # Camada de comunicação assíncrona e proxies
│   │   ├── types/            # Tipagens TypeScript completas
│   │   └── utils/            # Fallbacks de imagem SVG e utilitários
│   ├── vite.config.ts        # Configuração de proxy e network host 0.0.0.0
│   └── package.json
├── data/                     # Dados persistentes
│   ├── anime_catalog.json    # Catálogo com 960+ animes
│   ├── watch_history.json    # Histórico de avaliações do usuário
│   └── user_profile.json     # Vetor de preferências e estados
├── firebase.json             # Configurações de deploy para o Firebase Hosting
└── README.md
```

---

## 🚀 Como Executar Localmente

### 1. Pré-requisitos
* **Node.js** (v18+)
* **Python** (v3.10+)
* **Git**

### 2. Clonar o Repositório
```bash
git clone https://github.com/Gabriel-Tini/AnimeRecSys-AI.git
cd AnimeRecSys-AI
```

### 3. Configurar e Executar o Backend
```bash
# Criar e ativar ambiente virtual
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate

# Instalar dependências
pip install -r requirements.txt

# Iniciar o servidor FastAPI (Porta 8000)
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

### 4. Configurar e Executar o Frontend
Em outro terminal:
```bash
cd frontend

# Instalar dependências
npm install

# Iniciar o servidor de desenvolvimento Vite (Porta 5173)
npm run dev
```

* Acesse no navegador: `http://localhost:5173`
* Para acessar pelo celular ou TV na mesma rede Wi-Fi: `http://<SEU_IP_LOCAL>:5173`

---

## ☁️ Deploy no Firebase Hosting

O frontend React é 100% compatível com a infraestrutura global do Firebase Hosting:

```bash
# Compilar o frontend para produção
cd frontend
npm run build

# Na raiz do projeto, implantar no Firebase
cd ..
npx firebase deploy --only hosting
```

---

## 📄 Licença

Distribuído sob a licença **MIT**. Consulte `LICENSE` para mais informações.

Desenvolvido por **[Gabriel-Tini](https://github.com/Gabriel-Tini)**.
