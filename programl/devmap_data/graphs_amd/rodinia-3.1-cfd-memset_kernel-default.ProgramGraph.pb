

[external]
KcallBC
A
	full_text4
2
0%4 = tail call i64 @_Z13get_global_idj(i32 0) #2
4truncB+
)
	full_text

%5 = trunc i64 %4 to i32
"i64B

	full_text


i64 %4
3icmpB+
)
	full_text

%6 = icmp slt i32 %5, %2
"i32B

	full_text


i32 %5
6brB0
.
	full_text!

br i1 %6, label %7, label %12
 i1B

	full_text	

i1 %6
5trunc8B*
(
	full_text

%8 = trunc i16 %1 to i8
/shl8B&
$
	full_text

%9 = shl i64 %4, 32
$i648B

	full_text


i64 %4
8ashr8B.
,
	full_text

%10 = ashr exact i64 %9, 32
$i648B

	full_text


i64 %9
Vgetelementptr8BC
A
	full_text4
2
0%11 = getelementptr inbounds i8, i8* %0, i64 %10
%i648B

	full_text
	
i64 %10
Estore8B:
8
	full_text+
)
'store i8 %8, i8* %11, align 1, !tbaa !8
"i88B

	full_text	

i8 %8
%i8*8B

	full_text
	
i8* %11
'br8B

	full_text

br label %12
$ret8B

	full_text


ret void
$i328B

	full_text


i32 %2
$i168B

	full_text


i16 %1
$i8*8B

	full_text


i8* %0
-; undefined function B

	full_text

 
$i648B

	full_text


i64 32
#i328B

	full_text	

i32 0       	
 		               
	            	  "
memset_kernel"
_Z13get_global_idj*?
 rodinia-3.1-cfd-memset_kernel.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02?

wgsize
?

wgsize_log1p
I??A

transfer_bytes
?܎

devmap_label
 
 
transfer_bytes_log1p
I??A