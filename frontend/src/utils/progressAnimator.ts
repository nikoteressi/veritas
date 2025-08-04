/**
 * Progress Animation Controller
 * 
 * Provides smooth progress animations using requestAnimationFrame
 * with customizable easing functions and duration control.
 */

interface ProgressAnimatorOptions {
  duration?: number;
  easing?: (t: number) => number;
  updateInterval?: number;
  onUpdate?: (value: number) => void;
  onComplete?: (value: number) => void;
}

type EasingFunction = (t: number) => number;

class ProgressAnimator {
  private duration: number;
  private easing: EasingFunction;
  private onUpdate: (value: number) => void;
  private onComplete: (value: number) => void;
  
  private isAnimating: boolean;
  private animationId: number | null;
  private startTime: number | null;
  private startValue: number;
  private targetValue: number;
  private currentValue: number;

  constructor(options: ProgressAnimatorOptions = {}) {
    this.duration = options.duration || 300; // Default 300ms
    this.easing = options.easing || this.easeOutCubic;
    this.onUpdate = options.onUpdate || (() => {});
    this.onComplete = options.onComplete || (() => {});
    
    this.isAnimating = false;
    this.animationId = null;
    this.startTime = null;
    this.startValue = 0;
    this.targetValue = 0;
    this.currentValue = 0;
  }

  /**
   * Start animating from current value to target value
   */
  animateTo(targetValue: number, duration?: number): void {
    // Cancel any existing animation
    this.stop();
    
    this.startValue = this.currentValue;
    this.targetValue = Math.max(0, Math.min(100, targetValue));
    this.duration = duration || this.duration;
    this.startTime = performance.now();
    this.isAnimating = true;
    
    this._animate();
  }

  /**
   * Set progress immediately without animation
   */
  setImmediate(value: number): void {
    this.stop();
    this.currentValue = Math.max(0, Math.min(100, value));
    this.startValue = this.currentValue;
    this.targetValue = this.currentValue;
    this.onUpdate(this.currentValue);
  }

  /**
   * Stop current animation
   */
  stop(): void {
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }
    this.isAnimating = false;
  }

  /**
   * Get current progress value
   */
  getCurrentValue(): number {
    return this.currentValue;
  }

  /**
   * Check if animation is currently running
   */
  isRunning(): boolean {
    return this.isAnimating;
  }

  /**
   * Internal animation loop
   */
  private _animate(): void {
    if (!this.isAnimating) return;

    const currentTime = performance.now();
    const elapsed = currentTime - this.startTime!;
    const progress = Math.min(elapsed / this.duration, 1);

    // Apply easing function
    const easedProgress = this.easing(progress);
    
    // Calculate current value
    this.currentValue = this.startValue + (this.targetValue - this.startValue) * easedProgress;
    
    // Update callback
    this.onUpdate(this.currentValue);

    // Check if animation is complete
    if (progress >= 1) {
      this.currentValue = this.targetValue;
      this.isAnimating = false;
      this.onComplete(this.currentValue);
      return;
    }

    // Continue animation
    this.animationId = requestAnimationFrame(() => this._animate());
  }

  // Easing functions
  easeLinear(t: number): number {
    return t;
  }

  easeOutCubic(t: number): number {
    return 1 - Math.pow(1 - t, 3);
  }

  easeInOutCubic(t: number): number {
    return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
  }

  easeOutQuart(t: number): number {
    return 1 - Math.pow(1 - t, 4);
  }

  easeInOutQuart(t: number): number {
    return t < 0.5 ? 8 * t * t * t * t : 1 - Math.pow(-2 * t + 2, 4) / 2;
  }

  easeOutExpo(t: number): number {
    return t === 1 ? 1 : 1 - Math.pow(2, -10 * t);
  }

  easeOutBack(t: number): number {
    const c1 = 1.70158;
    const c3 = c1 + 1;
    return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
  }
}

// Easing functions as standalone utilities
const easingFunctions = {
  linear: (t: number): number => t,
  easeOutCubic: (t: number): number => 1 - Math.pow(1 - t, 3),
  easeInOutCubic: (t: number): number => t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2,
  easeOutQuart: (t: number): number => 1 - Math.pow(1 - t, 4),
  easeInOutQuart: (t: number): number => t < 0.5 ? 8 * t * t * t * t : 1 - Math.pow(-2 * t + 2, 4) / 2,
  easeOutExpo: (t: number): number => t === 1 ? 1 : 1 - Math.pow(2, -10 * t),
  easeOutBack: (t: number): number => {
    const c1 = 1.70158;
    const c3 = c1 + 1;
    return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
  }
};

/**
 * Factory function to create progress animator with predefined configurations
 */
export function createProgressAnimator(
  onUpdate: (value: number) => void, 
  onComplete?: (value: number) => void
): ProgressAnimator {
  return new ProgressAnimator({
    duration: 300,
    easing: easingFunctions.easeOutCubic,
    onUpdate,
    onComplete: onComplete || (() => {})
  });
}

// Additional factory methods for different animation types
createProgressAnimator.smooth = (onUpdate: (value: number) => void, onComplete?: (value: number) => void) => 
  new ProgressAnimator({
    duration: 300,
    easing: easingFunctions.easeOutCubic,
    onUpdate,
    onComplete: onComplete || (() => {})
  });

createProgressAnimator.fast = (onUpdate: (value: number) => void, onComplete?: (value: number) => void) => 
  new ProgressAnimator({
    duration: 150,
    easing: easingFunctions.easeOutQuart,
    onUpdate,
    onComplete: onComplete || (() => {})
  });

createProgressAnimator.slow = (onUpdate: (value: number) => void, onComplete?: (value: number) => void) => 
  new ProgressAnimator({
    duration: 600,
    easing: easingFunctions.easeInOutCubic,
    onUpdate,
    onComplete: onComplete || (() => {})
  });

createProgressAnimator.bouncy = (onUpdate: (value: number) => void, onComplete?: (value: number) => void) => 
  new ProgressAnimator({
    duration: 400,
    easing: easingFunctions.easeOutBack,
    onUpdate,
    onComplete: onComplete || (() => {})
  });

export default ProgressAnimator;