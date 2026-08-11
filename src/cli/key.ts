/**
 * Where the command line finds an upload key, and how it saves one.
 *
 * Three places, in order: the --key option, the environment, and a file the
 * `login` subcommand writes. The environment is what the page's copyable
 * command uses, since a key on the command line is visible to every user on
 * the machine through `ps` while another process's environment is not.
 */
import { homedir } from 'node:os';
import { join } from 'node:path';
import { mkdir, readFile, writeFile } from 'node:fs/promises';

export const KEY_ENV = 'TURING_SURFACE_CACHE_KEY';

const configDir = (): string =>
  join(process.env.XDG_CONFIG_HOME || join(homedir(), '.config'), 'turing-surface-cache');

export const keyPath = (): string => join(configDir(), 'key');

/** The saved key, or '' if there is none. */
export async function savedKey(): Promise<string> {
  try {
    return (await readFile(keyPath(), 'utf8')).trim();
  } catch {
    return '';
  }
}

/** Save a key for later runs, readable only by this user. */
export async function saveKey(key: string): Promise<string> {
  await mkdir(configDir(), { recursive: true, mode: 0o700 });
  const path = keyPath();
  await writeFile(path, `${key}\n`, { mode: 0o600 });
  return path;
}

/** --key, else the environment, else the saved key. */
export async function resolveKey(fromOption: string | undefined): Promise<string> {
  return (fromOption || process.env[KEY_ENV] || (await savedKey()) || '').trim();
}

/** Enough of a key to recognize it by, and no more. */
export const maskKey = (key: string): string =>
  key.length <= 4 ? '·'.repeat(key.length) : `····${key.slice(-4)}`;

/**
 * Read a secret from the terminal without echoing it. A piped stdin is read as
 * a plain line, so `echo $KEY | … login` works too.
 */
export async function promptSecret(prompt: string): Promise<string> {
  const stdin = process.stdin;
  process.stdout.write(prompt);
  if (!stdin.isTTY) {
    const chunks: Buffer[] = [];
    for await (const chunk of stdin) chunks.push(chunk as Buffer);
    process.stdout.write('\n');
    return Buffer.concat(chunks).toString('utf8').split('\n')[0].trim();
  }
  return new Promise<string>((resolve, reject) => {
    let typed = '';
    stdin.setRawMode(true);
    stdin.resume();
    stdin.setEncoding('utf8');
    const done = (finish: () => void): void => {
      stdin.setRawMode(false);
      stdin.pause();
      stdin.removeListener('data', onData);
      process.stdout.write('\n');
      finish();
    };
    const onData = (chunk: string): void => {
      for (const ch of chunk) {
        if (ch === '\r' || ch === '\n') return done(() => resolve(typed.trim()));
        if (ch === '\u0003') return done(() => reject(new Error('cancelled')));
        if (ch === '\u007f' || ch === '\b') typed = typed.slice(0, -1);
        else if (ch >= ' ') typed += ch;
      }
    };
    stdin.on('data', onData);
  });
}
