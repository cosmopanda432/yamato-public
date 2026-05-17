/**
 * ハルシネーション負例生成
 *
 * 標準入力に JSONL（1行 = {id, code}）を受け取り、各サンプルに対して
 * 4種類の mutation を適用し、変異後の候補を JSONL で出力する。
 *
 * Usage:
 *   echo '{"id":"1","code":"const arr = [1,2,3]; arr.push(4);"}' | node dist/mutate_for_hallucination.js
 *
 * Mutations:
 *   - fake_method:        メソッド呼び出しを存在しなさそうな名前に置換 (arr.push → arr.pushAndCheck)
 *   - drop_argument:      関数呼び出しから引数を1つ削除
 *   - add_argument:       関数呼び出しに引数を1つ追加
 *   - fake_import:        ファイル先頭に存在しない named import を追加
 *
 * 出力レコード:
 *   {
 *     parent_id: string,
 *     mutation: "fake_method" | "drop_argument" | "add_argument" | "fake_import",
 *     code: string,
 *     applied_at: {line, character} | null
 *   }
 *
 * 注意: ここでは tsc 検証は行わない。出力は tsc_strict_runner にパイプして
 * 「実際にコンパイル失敗するもののみ」を選別する。
 */

import { Project, SyntaxKind, Node, SourceFile, CallExpression } from "ts-morph";
import * as readline from "readline";

interface Input {
  id: string;
  code: string;
}

interface MutationRecord {
  parent_id: string;
  mutation: string;
  code: string;
  applied_at: { line: number; character: number } | null;
}

const FAKE_METHOD_SUFFIXES = ["AndCheck", "Sync2", "Fast", "X", "_unsafe"];
const FAKE_IMPORT_NAMES = ["__yamatoFakeApi", "__nonExistentSymbol", "__hirukoBait"];
const FAKE_IMPORT_MODULES = ["lodash", "fs/promises", "events"];

function pickPosition(sf: SourceFile, node: Node) {
  const start = node.getStart();
  return sf.getLineAndColumnAtPos(start);
}

function pickRandom<T>(items: T[]): T {
  return items[Math.floor(Math.random() * items.length)];
}

/** 最初の PropertyAccessExpression 呼び出しの method 名にサフィックスを付ける */
function fakeMethod(project: Project, code: string): MutationRecord | null {
  const sf = project.createSourceFile("m.ts", code, { overwrite: true });
  const calls = sf.getDescendantsOfKind(SyntaxKind.CallExpression);
  for (const call of calls) {
    const expr = call.getExpression();
    if (expr.getKind() === SyntaxKind.PropertyAccessExpression) {
      const pae = expr.asKindOrThrow(SyntaxKind.PropertyAccessExpression);
      const original = pae.getName();
      const fake = original + pickRandom(FAKE_METHOD_SUFFIXES);
      const pos = pickPosition(sf, pae.getNameNode());
      pae.getNameNode().replaceWithText(fake);
      return {
        parent_id: "",
        mutation: "fake_method",
        code: sf.getFullText(),
        applied_at: { line: pos.line, character: pos.column },
      };
    }
  }
  return null;
}

function dropArgument(project: Project, code: string): MutationRecord | null {
  const sf = project.createSourceFile("m.ts", code, { overwrite: true });
  const calls = sf.getDescendantsOfKind(SyntaxKind.CallExpression);
  for (const call of calls as CallExpression[]) {
    const args = call.getArguments();
    if (args.length >= 1) {
      const pos = pickPosition(sf, args[0]);
      call.removeArgument(0);
      return {
        parent_id: "",
        mutation: "drop_argument",
        code: sf.getFullText(),
        applied_at: { line: pos.line, character: pos.column },
      };
    }
  }
  return null;
}

function addArgument(project: Project, code: string): MutationRecord | null {
  const sf = project.createSourceFile("m.ts", code, { overwrite: true });
  const calls = sf.getDescendantsOfKind(SyntaxKind.CallExpression);
  for (const call of calls as CallExpression[]) {
    const pos = pickPosition(sf, call);
    // 既存引数の末尾に 0 を追加
    call.addArgument("0");
    return {
      parent_id: "",
      mutation: "add_argument",
      code: sf.getFullText(),
      applied_at: { line: pos.line, character: pos.column },
    };
  }
  return null;
}

function fakeImport(project: Project, code: string): MutationRecord {
  const sf = project.createSourceFile("m.ts", code, { overwrite: true });
  const fakeName = pickRandom(FAKE_IMPORT_NAMES);
  const fakeModule = pickRandom(FAKE_IMPORT_MODULES);
  sf.insertImportDeclaration(0, {
    namedImports: [fakeName],
    moduleSpecifier: fakeModule,
  });
  return {
    parent_id: "",
    mutation: "fake_import",
    code: sf.getFullText(),
    applied_at: { line: 1, character: 0 },
  };
}

async function main() {
  const project = new Project({
    useInMemoryFileSystem: true,
    compilerOptions: { target: 99 /* ES2022 */, module: 1 /* CommonJS */ },
  });

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

    const mutators: Array<(p: Project, c: string) => MutationRecord | null> = [
      fakeMethod,
      dropArgument,
      addArgument,
      fakeImport,
    ];

    for (const mut of mutators) {
      try {
        const rec = mut(project, input.code);
        if (rec === null) continue;
        rec.parent_id = input.id;
        process.stdout.write(JSON.stringify(rec) + "\n");
      } catch (e) {
        process.stderr.write(
          `mutate-error id=${input.id}: ${(e as Error).message}\n`,
        );
      }
    }
  }
}

main();
