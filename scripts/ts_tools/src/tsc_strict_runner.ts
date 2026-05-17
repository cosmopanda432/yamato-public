/**
 * tsc --strict ハーネス
 *
 * 標準入力に JSONL（1行 = {id, code}）を受け取り、各サンプルを
 * 単一ファイルとして strict コンパイルし、結果を JSONL で標準出力に流す。
 *
 * Usage:
 *   echo '{"id":"1","code":"const x: number = \"abc\";"}' | node dist/tsc_strict_runner.js
 *
 * 出力レコード:
 *   {
 *     id: string,
 *     ok: boolean,            // strict pass したか
 *     n_diagnostics: number,
 *     error_codes: number[],  // 例 [2322, 2339]
 *     messages: string[]      // 先頭 5 件
 *   }
 */

import * as ts from "typescript";
import * as readline from "readline";

const STRICT_OPTIONS: ts.CompilerOptions = {
  target: ts.ScriptTarget.ES2022,
  module: ts.ModuleKind.CommonJS,
  strict: true,
  noImplicitAny: true,
  strictNullChecks: true,
  noEmit: true,
  skipLibCheck: true,
  esModuleInterop: true,
};

interface Input {
  id: string;
  code: string;
  fileName?: string;
}

interface Output {
  id: string;
  ok: boolean;
  n_diagnostics: number;
  error_codes: number[];
  messages: string[];
}

function compileSingleFile(input: Input): Output {
  const fileName = input.fileName ?? "input.ts";
  const sourceFile = ts.createSourceFile(
    fileName,
    input.code,
    ts.ScriptTarget.ES2022,
    true,
    ts.ScriptKind.TS,
  );

  // メモリ上の compiler host: 入力ファイルと lib.d.ts だけ解決する
  const defaultHost = ts.createCompilerHost(STRICT_OPTIONS);
  const host: ts.CompilerHost = {
    ...defaultHost,
    getSourceFile: (name, languageVersion, onError) => {
      if (name === fileName) return sourceFile;
      return defaultHost.getSourceFile(name, languageVersion, onError);
    },
    fileExists: (name) => name === fileName || defaultHost.fileExists(name),
    readFile: (name) => (name === fileName ? input.code : defaultHost.readFile(name)),
    writeFile: () => {},
  };

  const program = ts.createProgram([fileName], STRICT_OPTIONS, host);
  const diagnostics = [
    ...program.getSyntacticDiagnostics(sourceFile),
    ...program.getSemanticDiagnostics(sourceFile),
  ];

  const errorCodes = diagnostics.map((d) => d.code);
  const messages = diagnostics
    .slice(0, 5)
    .map((d) => ts.flattenDiagnosticMessageText(d.messageText, "\n"));

  return {
    id: input.id,
    ok: diagnostics.length === 0,
    n_diagnostics: diagnostics.length,
    error_codes: errorCodes,
    messages,
  };
}

async function main() {
  const rl = readline.createInterface({
    input: process.stdin,
    crlfDelay: Infinity,
  });
  for await (const line of rl) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    let input: Input;
    try {
      input = JSON.parse(trimmed);
    } catch (e) {
      process.stderr.write(`parse-error: ${(e as Error).message}\n`);
      continue;
    }
    try {
      const out = compileSingleFile(input);
      process.stdout.write(JSON.stringify(out) + "\n");
    } catch (e) {
      const errOut: Output = {
        id: input.id,
        ok: false,
        n_diagnostics: -1,
        error_codes: [],
        messages: [`runner-error: ${(e as Error).message}`],
      };
      process.stdout.write(JSON.stringify(errOut) + "\n");
    }
  }
}

main();
