	.file	"pycann.c"
	.text
	.p2align 4,,15
.globl pycann_get_error
	.type	pycann_get_error, @function
pycann_get_error:
	pushl	%ebp
	movl	$pycann_error, %eax
	movl	%esp, %ebp
	popl	%ebp
	ret
	.size	pycann_get_error, .-pycann_get_error
	.p2align 4,,15
.globl pycann_reset_error
	.type	pycann_reset_error, @function
pycann_reset_error:
	pushl	%ebp
	movl	%esp, %ebp
	movb	$0, pycann_error
	popl	%ebp
	ret
	.size	pycann_reset_error, .-pycann_reset_error
	.p2align 4,,15
.globl pycann_get_memory_usage
	.type	pycann_get_memory_usage, @function
pycann_get_memory_usage:
	pushl	%ebp
	movl	%esp, %ebp
	movl	8(%ebp), %eax
	popl	%ebp
	movl	44(%eax), %eax
	ret
	.size	pycann_get_memory_usage, .-pycann_get_memory_usage
	.p2align 4,,15
.globl pycann_get_size
	.type	pycann_get_size, @function
pycann_get_size:
	pushl	%ebp
	movl	%esp, %ebp
	movl	8(%ebp), %eax
	popl	%ebp
	movl	(%eax), %eax
	ret
	.size	pycann_get_size, .-pycann_get_size
	.p2align 4,,15
.globl pycann_get_learning_rate
	.type	pycann_get_learning_rate, @function
pycann_get_learning_rate:
	pushl	%ebp
	movl	%esp, %ebp
	movl	8(%ebp), %eax
	popl	%ebp
	flds	4(%eax)
	ret
	.size	pycann_get_learning_rate, .-pycann_get_learning_rate
	.p2align 4,,15
.globl pycann_set_learning_rate
	.type	pycann_set_learning_rate, @function
pycann_set_learning_rate:
	pushl	%ebp
	movl	%esp, %ebp
	movl	12(%ebp), %edx
	movl	8(%ebp), %eax
	movl	%edx, 4(%eax)
	popl	%ebp
	ret
	.size	pycann_set_learning_rate, .-pycann_set_learning_rate
	.p2align 4,,15
.globl pycann_set_gamma
	.type	pycann_set_gamma, @function
pycann_set_gamma:
	pushl	%ebp
	movl	%esp, %ebp
	movl	12(%ebp), %eax
	cmpl	$3, %eax
	ja	.L15
	movl	8(%ebp), %edx
	flds	16(%ebp)
	fstps	8(%edx,%eax,4)
.L15:
	popl	%ebp
	ret
	.size	pycann_set_gamma, .-pycann_set_gamma
	.p2align 4,,15
.globl pycann_get_weight
	.type	pycann_get_weight, @function
pycann_get_weight:
	pushl	%ebp
	movl	%esp, %ebp
	movl	8(%ebp), %ecx
	movl	12(%ebp), %edx
	movl	(%ecx), %eax
	cmpl	%edx, %eax
	jbe	.L18
	cmpl	16(%ebp), %eax
	jbe	.L18
	imull	%eax, %edx
	movl	24(%ecx), %eax
	addl	16(%ebp), %edx
	popl	%ebp
	flds	(%eax,%edx,4)
	ret
	.p2align 4,,7
	.p2align 3
.L18:
	fldz
	popl	%ebp
	ret
	.size	pycann_get_weight, .-pycann_get_weight
	.p2align 4,,15
.globl pycann_set_weight
	.type	pycann_set_weight, @function
pycann_set_weight:
	pushl	%ebp
	movl	%esp, %ebp
	movl	8(%ebp), %ecx
	movl	12(%ebp), %edx
	movl	(%ecx), %eax
	cmpl	%edx, %eax
	jbe	.L23
	cmpl	16(%ebp), %eax
	jbe	.L23
	imull	%eax, %edx
	movl	24(%ecx), %eax
	addl	16(%ebp), %edx
	flds	20(%ebp)
	fstps	(%eax,%edx,4)
.L23:
	popl	%ebp
	ret
	.size	pycann_set_weight, .-pycann_set_weight
	.p2align 4,,15
.globl pycann_get_threshold
	.type	pycann_get_threshold, @function
pycann_get_threshold:
	pushl	%ebp
	movl	%esp, %ebp
	movl	8(%ebp), %eax
	movl	12(%ebp), %edx
	cmpl	%edx, (%eax)
	jbe	.L28
	movl	28(%eax), %eax
	popl	%ebp
	flds	(%eax,%edx,4)
	ret
	.p2align 4,,7
	.p2align 3
.L28:
	fldz
	popl	%ebp
	ret
	.size	pycann_get_threshold, .-pycann_get_threshold
	.p2align 4,,15
.globl pycann_set_threshold
	.type	pycann_set_threshold, @function
pycann_set_threshold:
	pushl	%ebp
	movl	%esp, %ebp
	movl	8(%ebp), %eax
	movl	12(%ebp), %edx
	cmpl	%edx, (%eax)
	jbe	.L31
	movl	28(%eax), %eax
	flds	16(%ebp)
	fstps	(%eax,%edx,4)
.L31:
	popl	%ebp
	ret
	.size	pycann_set_threshold, .-pycann_set_threshold
	.p2align 4,,15
.globl pycann_get_activation
	.type	pycann_get_activation, @function
pycann_get_activation:
	pushl	%ebp
	movl	%esp, %ebp
	movl	8(%ebp), %eax
	movl	12(%ebp), %edx
	cmpl	%edx, (%eax)
	jbe	.L36
	movl	32(%eax), %eax
	popl	%ebp
	flds	(%eax,%edx,4)
	ret
	.p2align 4,,7
	.p2align 3
.L36:
	fldz
	popl	%ebp
	ret
	.size	pycann_get_activation, .-pycann_get_activation
	.p2align 4,,15
.globl pycann_set_activation
	.type	pycann_set_activation, @function
pycann_set_activation:
	pushl	%ebp
	movl	%esp, %ebp
	movl	8(%ebp), %eax
	movl	12(%ebp), %edx
	cmpl	%edx, (%eax)
	jbe	.L39
	movl	32(%eax), %eax
	flds	16(%ebp)
	fstps	(%eax,%edx,4)
.L39:
	popl	%ebp
	ret
	.size	pycann_set_activation, .-pycann_set_activation
	.p2align 4,,15
.globl pycann_get_mod_neuron
	.type	pycann_get_mod_neuron, @function
pycann_get_mod_neuron:
	pushl	%ebp
	xorl	%eax, %eax
	movl	%esp, %ebp
	popl	%ebp
	ret
	.size	pycann_get_mod_neuron, .-pycann_get_mod_neuron
	.p2align 4,,15
.globl pycann_get_mod_weight
	.type	pycann_get_mod_weight, @function
pycann_get_mod_weight:
	pushl	%ebp
	movl	%esp, %ebp
	movl	8(%ebp), %eax
	movl	12(%ebp), %edx
	cmpl	%edx, (%eax)
	jbe	.L46
	movl	36(%eax), %eax
	popl	%ebp
	flds	(%eax,%edx,4)
	ret
	.p2align 4,,7
	.p2align 3
.L46:
	fldz
	popl	%ebp
	ret
	.size	pycann_get_mod_weight, .-pycann_get_mod_weight
	.p2align 4,,15
.globl pycann_set_mod
	.type	pycann_set_mod, @function
pycann_set_mod:
	pushl	%ebp
	movl	%esp, %ebp
	movl	8(%ebp), %ecx
	pushl	%ebx
	movl	12(%ebp), %ebx
	flds	20(%ebp)
	movl	(%ecx), %eax
	cmpl	%ebx, %eax
	jbe	.L50
	cmpl	16(%ebp), %eax
	jbe	.L51
	movl	16(%ebp), %edx
	movl	40(%ecx), %eax
	fmuls	4(%ecx)
	sall	$2, %edx
	addl	32(%ecx), %edx
	movl	%edx, (%eax,%ebx,4)
	movl	36(%ecx), %eax
	fstps	(%eax,%ebx,4)
	jmp	.L49
	.p2align 4,,7
	.p2align 3
.L50:
	fstp	%st(0)
	jmp	.L49
	.p2align 4,,7
	.p2align 3
.L51:
	fstp	%st(0)
.L49:
	popl	%ebx
	popl	%ebp
	ret
	.size	pycann_set_mod, .-pycann_set_mod
	.p2align 4,,15
.globl pycann_get_outputs
	.type	pycann_get_outputs, @function
pycann_get_outputs:
	pushl	%ebp
	movl	%esp, %ebp
	pushl	%edi
	movl	8(%ebp), %edi
	pushl	%esi
	pushl	%ebx
	movl	(%edi), %eax
	movl	%eax, %ebx
	subl	56(%edi), %ebx
	cmpl	%ebx, %eax
	jbe	.L55
	movl	12(%ebp), %edx
	leal	0(,%ebx,4), %ecx
	movl	32(%edi), %esi
	addl	%ecx, %edx
	.p2align 4,,7
	.p2align 3
.L54:
	movl	(%esi,%ecx), %eax
	addl	$1, %ebx
	addl	$4, %ecx
	movl	%eax, (%edx)
	addl	$4, %edx
	cmpl	%ebx, (%edi)
	ja	.L54
.L55:
	popl	%ebx
	popl	%esi
	popl	%edi
	popl	%ebp
	ret
	.size	pycann_get_outputs, .-pycann_get_outputs
	.p2align 4,,15
.globl pycann_step
	.type	pycann_step, @function
pycann_step:
	pushl	%ebp
	movl	%esp, %ebp
	pushl	%edi
	pushl	%esi
	pushl	%ebx
	subl	$24, %esp
	movl	12(%ebp), %ecx
	movl	8(%ebp), %edi
	testl	%ecx, %ecx
	je	.L73
	movl	(%edi), %eax
	movl	$0, -16(%ebp)
	movl	%eax, -36(%ebp)
	.p2align 4,,7
	.p2align 3
.L59:
	movl	-36(%ebp), %eax
	testl	%eax, %eax
	je	.L72
	movl	32(%edi), %edx
	xorl	%esi, %esi
	movl	28(%edi), %ebx
	movl	%edx, -32(%ebp)
	movl	%ebx, -28(%ebp)
	jmp	.L70
	.p2align 4,,7
	.p2align 3
.L76:
	movl	52(%edi), %eax
	flds	(%eax,%esi,4)
.L61:
	movl	-28(%ebp), %edx
	movl	-24(%ebp), %eax
	fcomps	(%edx,%eax)
	fnstsw	%ax
	sahf
	ja	.L68
	movl	-20(%ebp), %ebx
	addl	$1, %esi
	movl	$0x3dcccccd, %eax
	cmpl	%esi, -36(%ebp)
	movl	%eax, (%ebx)
	jbe	.L72
.L70:
	movl	-32(%ebp), %edx
	leal	0(,%esi,4), %eax
	movl	%eax, -24(%ebp)
	addl	%eax, %edx
	cmpl	%esi, 48(%edi)
	movl	%edx, -20(%ebp)
	ja	.L76
	movl	36(%edi), %eax
	flds	(%edx)
	flds	(%eax,%esi,4)
	ftst
	fnstsw	%ax
	sahf
	jne	.L62
	fstp	%st(0)
	movl	-36(%ebp), %edx
	fldz
	testl	%edx, %edx
	je	.L78
.L64:
	ftst
	fnstsw	%ax
	sahf
	jne	.L65
	fstp	%st(0)
	fstp	%st(0)
	movl	-36(%ebp), %ecx
	xorl	%edx, %edx
	movl	24(%edi), %ebx
	fldz
	imull	%esi, %ecx
	.p2align 4,,7
	.p2align 3
.L66:
	leal	(%ecx,%edx), %eax
	flds	(%ebx,%eax,4)
	movl	-32(%ebp), %eax
	fmuls	(%eax,%edx,4)
	addl	$1, %edx
	cmpl	%edx, -36(%ebp)
	faddp	%st, %st(1)
	ja	.L66
	jmp	.L61
	.p2align 4,,7
	.p2align 3
.L68:
	movl	-20(%ebp), %ebx
	addl	$1, %esi
	movl	$0x3f800000, %eax
	cmpl	%esi, -36(%ebp)
	movl	%eax, (%ebx)
	ja	.L70
.L72:
	addl	$1, -16(%ebp)
	movl	-16(%ebp), %eax
	cmpl	%eax, 12(%ebp)
	ja	.L59
.L73:
	addl	$24, %esp
	popl	%ebx
	popl	%esi
	popl	%edi
	popl	%ebp
	ret
	.p2align 4,,7
	.p2align 3
.L62:
	movl	-24(%ebp), %edx
	movl	40(%edi), %eax
	movl	(%eax,%edx), %eax
	movl	-36(%ebp), %edx
	flds	(%eax)
	testl	%edx, %edx
	fmulp	%st, %st(1)
	jne	.L64
	fstp	%st(0)
	fstp	%st(0)
	jmp	.L77
.L78:
	fstp	%st(0)
	fstp	%st(0)
.L77:
	fldz
	jmp	.L61
	.p2align 4,,7
	.p2align 3
.L65:
	movl	-36(%ebp), %ecx
	xorl	%edx, %edx
	movl	-32(%ebp), %ebx
	fldz
	imull	%esi, %ecx
	.p2align 4,,7
	.p2align 3
.L67:
	leal	(%ecx,%edx), %eax
	sall	$2, %eax
	addl	24(%edi), %eax
	flds	(%eax)
	flds	(%ebx,%edx,4)
	addl	$1, %edx
	fld	%st(1)
	fmul	%st(1), %st
	cmpl	-36(%ebp), %edx
	faddp	%st, %st(3)
	flds	12(%edi)
	fmul	%st(5), %st
	fadds	24(%edi)
	flds	8(%edi)
	fmul	%st(6), %st
	fmul	%st(2), %st
	faddp	%st, %st(1)
	fxch	%st(1)
	fmuls	16(%edi)
	faddp	%st, %st(1)
	fmul	%st(3), %st
	faddp	%st, %st(1)
	fstps	(%eax)
	jb	.L67
	fstp	%st(1)
	fstp	%st(1)
	jmp	.L61
	.size	pycann_step, .-pycann_step
	.p2align 4,,15
.globl pycann_set_inputs
	.type	pycann_set_inputs, @function
pycann_set_inputs:
	pushl	%ebp
	movl	%esp, %ebp
	subl	$24, %esp
	movl	8(%ebp), %eax
	movl	52(%eax), %edx
	movl	48(%eax), %eax
	movl	%edx, (%esp)
	sall	$2, %eax
	movl	%eax, 8(%esp)
	movl	12(%ebp), %eax
	movl	%eax, 4(%esp)
	call	memcpy
	leave
	ret
	.size	pycann_set_inputs, .-pycann_set_inputs
	.p2align 4,,15
	.type	pycann_set_error, @function
pycann_set_error:
	pushl	%ebp
	movl	%esp, %ebp
	subl	$24, %esp
	leal	12(%ebp), %eax
	movl	%eax, 12(%esp)
	movl	8(%ebp), %eax
	movl	$1024, 4(%esp)
	movl	$pycann_error, (%esp)
	movl	%eax, 8(%esp)
	call	vsnprintf
	leave
	ret
	.size	pycann_set_error, .-pycann_set_error
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC4:
	.string	"Invalid gamma index: %d"
	.text
	.p2align 4,,15
.globl pycann_get_gamma
	.type	pycann_get_gamma, @function
pycann_get_gamma:
	pushl	%ebp
	movl	%esp, %ebp
	subl	$8, %esp
	movl	12(%ebp), %eax
	cmpl	$3, %eax
	ja	.L84
	movl	8(%ebp), %edx
	flds	8(%edx,%eax,4)
	leave
	ret
	.p2align 4,,7
	.p2align 3
.L84:
	movl	%eax, 4(%esp)
	movl	$.LC4, (%esp)
	call	pycann_set_error
	leave
	fldz
	ret
	.size	pycann_get_gamma, .-pycann_get_gamma
	.p2align 4,,15
.globl pycann_del
	.type	pycann_del, @function
pycann_del:
	pushl	%ebp
	movl	%esp, %ebp
	pushl	%ebx
	subl	$4, %esp
	movl	8(%ebp), %ebx
	movl	24(%ebx), %eax
	movl	%eax, (%esp)
	call	free
	movl	28(%ebx), %eax
	movl	%eax, (%esp)
	call	free
	movl	32(%ebx), %eax
	movl	%eax, (%esp)
	call	free
	movl	%ebx, 8(%ebp)
	addl	$4, %esp
	popl	%ebx
	popl	%ebp
	jmp	free
	.size	pycann_del, .-pycann_del
	.p2align 4,,15
.globl pycann_malloc
	.type	pycann_malloc, @function
pycann_malloc:
	pushl	%ebp
	movl	%esp, %ebp
	movl	8(%ebp), %eax
	movl	12(%ebp), %edx
	addl	%edx, 44(%eax)
	movl	%edx, 8(%ebp)
	popl	%ebp
	jmp	malloc
	.size	pycann_malloc, .-pycann_malloc
	.p2align 4,,15
.globl pycann_new
	.type	pycann_new, @function
pycann_new:
	pushl	%ebp
	movl	%esp, %ebp
	pushl	%edi
	pushl	%esi
	pushl	%ebx
	subl	$28, %esp
	movl	8(%ebp), %edi
	movl	$60, (%esp)
	call	malloc
	leal	0(,%edi,4), %ebx
	movl	%eax, %esi
	movl	%edi, %eax
	imull	%edi, %eax
	sall	$2, %eax
	leal	60(%eax), %edx
	movl	%edx, 44(%esi)
	movl	%eax, (%esp)
	call	malloc
	addl	%ebx, 44(%esi)
	movl	%ebx, (%esp)
	movl	%eax, 24(%esi)
	call	malloc
	addl	%ebx, 44(%esi)
	movl	%ebx, (%esp)
	movl	%eax, 28(%esi)
	call	malloc
	addl	%ebx, 44(%esi)
	movl	%ebx, (%esp)
	movl	%eax, 32(%esi)
	call	malloc
	addl	%ebx, 44(%esi)
	movl	%ebx, (%esp)
	movl	%eax, 40(%esi)
	call	malloc
	movl	%eax, 36(%esi)
	movl	16(%ebp), %eax
	sall	$2, %eax
	movl	%eax, (%esp)
	addl	%eax, 44(%esi)
	call	malloc
	movl	16(%ebp), %ecx
	movl	%edi, (%esi)
	movl	%ecx, 56(%esi)
	movl	%eax, 52(%esi)
	xorl	%eax, %eax
	testl	%edi, %edi
	movl	%eax, 4(%esi)
	movl	%eax, 8(%esi)
	movl	%eax, 12(%esi)
	movl	%eax, 16(%esi)
	movl	%eax, 20(%esi)
	movl	12(%ebp), %eax
	movl	%eax, 48(%esi)
	je	.L92
	movl	24(%esi), %eax
	xorl	%ebx, %ebx
	movl	28(%esi), %ecx
	movl	%eax, -32(%ebp)
	movl	32(%esi), %eax
	movl	%ecx, -28(%ebp)
	movl	40(%esi), %ecx
	movl	%eax, -24(%ebp)
	movl	36(%esi), %eax
	movl	%ecx, -20(%ebp)
	movl	%eax, -16(%ebp)
	.p2align 4,,7
	.p2align 3
.L93:
	movl	%edi, %eax
	movl	-32(%ebp), %ecx
	imull	%ebx, %eax
	leal	(%ecx,%eax,4), %edx
	xorl	%eax, %eax
	xorl	%ecx, %ecx
	.p2align 4,,7
	.p2align 3
.L94:
	addl	$1, %eax
	movl	%ecx, (%edx)
	addl	$4, %edx
	cmpl	%eax, %edi
	ja	.L94
	movl	-28(%ebp), %eax
	movl	%ecx, (%eax,%ebx,4)
	movl	-24(%ebp), %eax
	movl	%ecx, (%eax,%ebx,4)
	movl	-20(%ebp), %eax
	movl	$0, (%eax,%ebx,4)
	movl	-16(%ebp), %eax
	movl	%ecx, (%eax,%ebx,4)
	addl	$1, %ebx
	cmpl	%ebx, %edi
	ja	.L93
.L92:
	movl	12(%ebp), %ebx
	testl	%ebx, %ebx
	je	.L95
	movl	52(%esi), %edx
	xorl	%eax, %eax
	.p2align 4,,7
	.p2align 3
.L96:
	movl	$0x00000000, (%edx,%eax,4)
	addl	$1, %eax
	cmpl	%eax, 12(%ebp)
	ja	.L96
.L95:
	addl	$28, %esp
	movl	%esi, %eax
	popl	%ebx
	popl	%esi
	popl	%edi
	popl	%ebp
	ret
	.size	pycann_new, .-pycann_new
	.local	pycann_error
	.comm	pycann_error,1024,32
	.ident	"GCC: (SUSE Linux) 4.3.1 20080507 (prerelease) [gcc-4_3-branch revision 135036]"
	.section	.note.GNU-stack,"",@progbits
