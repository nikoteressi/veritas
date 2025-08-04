/**
 * Utility types for enhanced type safety
 */

// Utility type to make all properties required
export type RequiredFields<T, K extends keyof T> = T & Required<Pick<T, K>>;

// Utility type to make specific properties optional
export type OptionalFields<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;

// Utility type for non-empty arrays
export type NonEmptyArray<T> = [T, ...T[]];

// Utility type for branded types (nominal typing)
export type Brand<T, B> = T & { __brand: B };

// Common branded types
export type SessionId = Brand<string, 'SessionId'>;
export type StepId = Brand<string, 'StepId'>;
export type FileId = Brand<string, 'FileId'>;
export type UserId = Brand<string, 'UserId'>;

// Utility type for exhaustive checking
export function assertNever(value: never): never {
  throw new Error(`Unexpected value: ${value}`);
}

// Type guard utilities
export function isString(value: unknown): value is string {
  return typeof value === 'string';
}

export function isNumber(value: unknown): value is number {
  return typeof value === 'number' && !isNaN(value);
}

export function isBoolean(value: unknown): value is boolean {
  return typeof value === 'boolean';
}

export function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

export function isArray<T>(value: unknown): value is T[] {
  return Array.isArray(value);
}

export function isNonEmptyArray<T>(value: T[]): value is NonEmptyArray<T> {
  return value.length > 0;
}

// Utility type for deep readonly
export type DeepReadonly<T> = {
  readonly [P in keyof T]: T[P] extends (infer U)[]
    ? DeepReadonlyArray<U>
    : T[P] extends object
    ? DeepReadonly<T[P]>
    : T[P];
};

export interface DeepReadonlyArray<T> extends ReadonlyArray<DeepReadonly<T>> {}

// Utility type for deep partial
export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends (infer U)[]
    ? DeepPartial<U>[]
    : T[P] extends object
    ? DeepPartial<T[P]>
    : T[P];
};

// Utility type for strict object keys
export type StrictKeys<T> = keyof T;

// Utility type for value of object
export type ValueOf<T> = T[keyof T];

// Utility type for function parameters
export type Parameters<T extends (...args: any[]) => any> = T extends (...args: infer P) => any ? P : never;

// Utility type for function return type
export type ReturnType<T extends (...args: any[]) => any> = T extends (...args: any[]) => infer R ? R : any;

// Utility type for promise resolution
export type Awaited<T> = T extends PromiseLike<infer U> ? U : T;

// Utility type for conditional types
export type If<C extends boolean, T, F> = C extends true ? T : F;

// Utility type for union to intersection
export type UnionToIntersection<U> = (U extends any ? (k: U) => void : never) extends (k: infer I) => void ? I : never;

// Utility type for tuple to union
export type TupleToUnion<T extends readonly unknown[]> = T[number];

// Utility type for safe array access
export type SafeArrayAccess<T extends readonly unknown[], K extends keyof T> = T[K];

// Utility type for environment variables
export type EnvVar<T extends string> = Brand<string, `EnvVar_${T}`>;

// Result type for error handling
export type Result<T, E = Error> = 
  | { success: true; data: T }
  | { success: false; error: E };

// Option type for nullable values
export type Option<T> = T | null | undefined;

// Utility function to create a result
export function createResult<T, E = Error>(
  data?: T,
  error?: E
): Result<T, E> {
  if (error) {
    return { success: false, error };
  }
  if (data !== undefined) {
    return { success: true, data };
  }
  throw new Error('Either data or error must be provided');
}

// Utility function to handle options
export function isSome<T>(value: Option<T>): value is T {
  return value !== null && value !== undefined;
}

export function isNone<T>(value: Option<T>): value is null | undefined {
  return value === null || value === undefined;
}