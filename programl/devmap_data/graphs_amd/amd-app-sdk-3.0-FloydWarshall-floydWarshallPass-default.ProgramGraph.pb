

[external]
KcallBC
A
	full_text4
2
0%5 = tail call i64 @_Z13get_global_idj(i32 0) #2
4truncB+
)
	full_text

%6 = trunc i64 %5 to i32
"i64B

	full_text


i64 %5
KcallBC
A
	full_text4
2
0%7 = tail call i64 @_Z13get_global_idj(i32 1) #2
4truncB+
)
	full_text

%8 = trunc i64 %7 to i32
"i64B

	full_text


i64 %7
-mulB&
$
	full_text

%9 = mul i32 %8, %2
"i32B

	full_text


i32 %8
.addB'
%
	full_text

%10 = add i32 %9, %6
"i32B

	full_text


i32 %9
"i32B

	full_text


i32 %6
4zextB,
*
	full_text

%11 = zext i32 %10 to i64
#i32B

	full_text
	
i32 %10
VgetelementptrBE
C
	full_text6
4
2%12 = getelementptr inbounds i32, i32* %0, i64 %11
#i64B

	full_text
	
i64 %11
FloadB>
<
	full_text/
-
+%13 = load i32, i32* %12, align 4, !tbaa !8
%i32*B

	full_text


i32* %12
.addB'
%
	full_text

%14 = add i32 %9, %3
"i32B

	full_text


i32 %9
4zextB,
*
	full_text

%15 = zext i32 %14 to i64
#i32B

	full_text
	
i32 %14
VgetelementptrBE
C
	full_text6
4
2%16 = getelementptr inbounds i32, i32* %0, i64 %15
#i64B

	full_text
	
i64 %15
FloadB>
<
	full_text/
-
+%17 = load i32, i32* %16, align 4, !tbaa !8
%i32*B

	full_text


i32* %16
.mulB'
%
	full_text

%18 = mul i32 %3, %2
/addB(
&
	full_text

%19 = add i32 %18, %6
#i32B

	full_text
	
i32 %18
"i32B

	full_text


i32 %6
4zextB,
*
	full_text

%20 = zext i32 %19 to i64
#i32B

	full_text
	
i32 %19
VgetelementptrBE
C
	full_text6
4
2%21 = getelementptr inbounds i32, i32* %0, i64 %20
#i64B

	full_text
	
i64 %20
FloadB>
<
	full_text/
-
+%22 = load i32, i32* %21, align 4, !tbaa !8
%i32*B

	full_text


i32* %21
0addB)
'
	full_text

%23 = add i32 %22, %17
#i32B

	full_text
	
i32 %22
#i32B

	full_text
	
i32 %17
6icmpB.
,
	full_text

%24 = icmp slt i32 %23, %13
#i32B

	full_text
	
i32 %23
#i32B

	full_text
	
i32 %13
8brB2
0
	full_text#
!
br i1 %24, label %25, label %27
!i1B

	full_text


i1 %24
Hstore8B=
;
	full_text.
,
*store i32 %23, i32* %12, align 4, !tbaa !8
%i328B

	full_text
	
i32 %23
'i32*8B

	full_text


i32* %12
Xgetelementptr8BE
C
	full_text6
4
2%26 = getelementptr inbounds i32, i32* %1, i64 %11
%i648B

	full_text
	
i64 %11
Gstore8B<
:
	full_text-
+
)store i32 %3, i32* %26, align 4, !tbaa !8
'i32*8B

	full_text


i32* %26
'br8B

	full_text

br label %27
$ret8B

	full_text


ret void
&i32*8B

	full_text
	
i32* %0
&i32*8B

	full_text
	
i32* %1
$i328B

	full_text


i32 %2
$i328B

	full_text


i32 %3
-; undefined function B

	full_text

 
#i328B

	full_text	

i32 0
#i328B

	full_text	

i32 1        	
 	 		                      !    "# "" $% $& $$ '( ') '' *+ *- ,. ,, /0 // 12 11 35 5 5  6 /7 7 8 8 8 1    
 	           !  #" % &$ ( )' +$ - . 0/ 2* ,* 43 4 4 99 99  99 : ; "
floydWarshallPass"
_Z13get_global_idj*?
FloydWarshall_Kernels.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282?

devmap_label
 

transfer_bytes
?? 

wgsize
?
 
transfer_bytes_log1p
~?RA

wgsize_log1p
~?RA